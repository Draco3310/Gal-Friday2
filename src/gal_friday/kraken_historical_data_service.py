"""Concrete implementation of HistoricalDataService for Kraken exchange."""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union, cast

import pandas as pd
import pandas_ta as ta
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from .historical_data_service import HistoricalDataService
from .logger_service import LoggerService


class RateLimitTracker:
    """Tracks and manages API rate limits."""

    def __init__(self, tier: str = "default", logger: Optional[LoggerService] = None):
        """Initialize rate limit tracker with specified tier settings.

        Args:
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
                self.logger.debug(f"Rate limit: Waiting {wait_time:.3f}s before next request")
                await asyncio.sleep(wait_time)

        self.last_request_time = datetime.now()


class CircuitBreaker:
    """Implements circuit breaker pattern for API calls."""

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_timeout: int = 60,
        logger: Optional[LoggerService] = None,
    ):
        """Initialize circuit breaker with specified thresholds.

        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            reset_timeout: Time in seconds before attempting to close circuit after failure
            logger: Logger service for logging circuit state changes
        """
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout  # seconds
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.last_failure_time: Optional[datetime] = None
        self.logger = logger or logging.getLogger(__name__)

    async def execute(
        self, func: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any
    ) -> Any:
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
                raise Exception("Circuit breaker is OPEN - API calls blocked")

        try:
            result = await func(*args, **kwargs)

            # Success - reset circuit breaker if in HALF-OPEN
            if self.state == "HALF-OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                self.logger.info("Circuit breaker state changed to CLOSED")

            return result

        except Exception as e:
            # Failure - increment counter and potentially open circuit
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == "HALF-OPEN" or self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.warning(
                    f"Circuit breaker state changed to OPEN after {self.failure_count} failures"
                )

            raise e


class KrakenHistoricalDataService(HistoricalDataService):
    """Kraken implementation of the HistoricalDataService interface."""

    def __init__(self, config: Dict[str, Any], logger_service: LoggerService):
        """
        Initialize the Kraken historical data service.

        Args:
            config: Configuration dict with settings for API and storage
            logger_service: Logger service for logging
        """
        self.config = config
        self.logger = logger_service
        self._source_module = self.__class__.__name__

        # Initialize InfluxDB client
        influx_config = config.get("influxdb", {})
        self.influx_client = InfluxDBClient(
            url=influx_config.get("url", "http://localhost:8086"),
            token=influx_config.get("token", ""),
            org=influx_config.get("org", "gal_friday"),
            debug=config.get("debug", False),
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.influx_client.query_api()

        # Measurement settings
        self.ohlcv_measurement = "market_data_ohlcv"
        self.trades_measurement = "market_data_trades"

        # Initialize rate limiter
        self.rate_limiter = RateLimitTracker(
            tier=config.get("api_tier", "default"), logger=logger_service
        )

        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get("failure_threshold", 3),
            reset_timeout=config.get("reset_timeout", 60),
            logger=logger_service,
        )

        # Cache for frequently accessed data
        self._ohlcv_cache: Dict[Any, pd.DataFrame] = {}  # Simple cache for OHLCV data

        self.logger.info(
            "KrakenHistoricalDataService initialized", source_module=self._source_module
        )

    async def get_historical_ohlcv(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str,  # e.g., "1m", "5m", "1h"
    ) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data for a given pair, time range, and interval.

        This method first checks InfluxDB for stored data, then fetches any missing
        data from the Kraken API.
        """
        self.logger.info(
            f"Getting historical OHLCV for {trading_pair}from {start_time} to {end_time} "
            f"(interval: {interval})",
            source_module=self._source_module,
        )

        # Check if data already exists in InfluxDB
        df = await self._query_ohlcv_data_from_influxdb(
            trading_pair, start_time, end_time, interval
        )

        # If complete data is available in InfluxDB, return it
        if df is not None and not df.empty:
            available_start = df.index.min()
            available_end = df.index.max()

            # If we have all the data needed
            if available_start <= start_time and available_end >= end_time:
                self.logger.info(
                    f"Complete OHLCV data found in InfluxDB for {trading_pair}",
                    source_module=self._source_module,
                )
                # Slice to exact range and return
                return df[(df.index >= start_time) & (df.index <= end_time)]

        # Determine what data is missing
        missing_ranges = self._get_missing_ranges(df, start_time, end_time)
        if not missing_ranges:
            # Should have data but is empty
            self.logger.warning(
                f"No missing ranges but data is empty for {trading_pair}",
                source_module=self._source_module,
            )
            return None

        # Fetch missing data from Kraken API
        complete_df = df if df is not None and not df.empty else None

        for range_start, range_end in missing_ranges:
            self.logger.info(
                f"Fetching missing data for {trading_pair} from {range_start} to {range_end}",
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
                        f"Validation failed for {trading_pair} OHLCV data",
                        source_module=self._source_module,
                    )
            else:
                self.logger.warning(
                    f"No data returned from API for {trading_pair} between {range_start} "
                    f"and {range_end}",
                    source_module=self._source_module,
                )

        # Final results
        if complete_df is not None and not complete_df.empty:
            # Slice to exact requested range
            result_df = complete_df[
                (complete_df.index >= start_time) & (complete_df.index <= end_time)
            ]
            self.logger.info(
                f"Returning {len(result_df)} OHLCV rows for {trading_pair}",
                source_module=self._source_module,
            )
            return result_df
        else:
            self.logger.warning(
                f"No OHLCV data available for {trading_pair}", source_module=self._source_module
            )
            return None

    async def get_historical_trades(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Optional[pd.DataFrame]:
        """Get historical trade data for a given pair and time range."""
        self.logger.info(
            f"Getting historical trades for {trading_pair} from {start_time} to {end_time}",
            source_module=self._source_module,
        )

        # First check InfluxDB for stored trade data
        df = await self._query_trades_data_from_influxdb(trading_pair, start_time, end_time)

        # If complete data is available in InfluxDB, return it
        if df is not None and not df.empty:
            available_start = df.index.min()
            available_end = df.index.max()

            # If we have all the data needed
            if available_start <= start_time and available_end >= end_time:
                self.logger.info(
                    f"Complete trade data found in InfluxDB for {trading_pair}",
                    source_module=self._source_module,
                )
                # Slice to exact range and return
                return df[(df.index >= start_time) & (df.index <= end_time)]

        # TODO: Implement fetching trade data from Kraken API
        # This would be similar to _fetch_ohlcv_data but for trades

        self.logger.warning(
            f"Trade data fetching not fully implemented for {trading_pair}",
            source_module=self._source_module,
        )
        return df

    def get_next_bar(self, trading_pair: str, timestamp: datetime) -> Optional[pd.Series]:
        """Get the next available OHLCV bar after the given timestamp."""
        # Convert to synchronous query for this method
        query = f"""
        from(bucket: "{self.config.get("influxdb", {}).get("bucket", "gal_friday")}")
          |> range(start: {timestamp.isoformat()}, stop: {(
            timestamp + timedelta(days=7)).isoformat()})
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
                    f"No next bar found for {trading_pair} after {timestamp}",
                    source_module=self._source_module,
                )
                return None

            # Convert to pandas Series
            for table in tables:
                for record in table.records:
                    series = pd.Series(
                        {
                            "open": float(record.values.get("open", 0)),
                            "high": float(record.values.get("high", 0)),
                            "low": float(record.values.get("low", 0)),
                            "close": float(record.values.get("close", 0)),
                            "volume": float(record.values.get("volume", 0)),
                        },
                        name=record.values.get("_time"),
                    )
                    return series

            return None

        except Exception as e:
            self.logger.error(
                f"Error getting next bar for {trading_pair}: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
            return None

    def get_atr(
        self, trading_pair: str, timestamp: datetime, period: int = 14
    ) -> Optional[Decimal]:
        """Get the Average True Range indicator value at the given timestamp."""
        # Need to get enough bars before the timestamp for ATR calculation
        query = f"""
        from(bucket: "{self.config.get("influxdb", {}).get("bucket", "gal_friday")}")
          |> range(start: {(
            timestamp - timedelta(days=30)).isoformat()}, stop: {timestamp.isoformat()})
          |> filter(fn: (r) => r["_measurement"] == "{self.ohlcv_measurement}")
          |> filter(fn: (r) => r["trading_pair"] == "{trading_pair}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"], desc: false)
        """

        try:
            tables = self.query_api.query(query)
            if not tables:
                self.logger.debug(
                    f"No data found for ATR calculation for {trading_pair}",
                    source_module=self._source_module,
                )
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
                        }
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

            return None

        except Exception as e:
            self.logger.error(
                f"Error calculating ATR for {trading_pair}: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
            return None

    async def _fetch_ohlcv_data(
        self, trading_pair: str, start_time: datetime, end_time: datetime, interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Kraken API."""
        try:
            # Wait for rate limit if needed
            await self.rate_limiter.wait_if_needed()

            # Execute API call with circuit breaker
            # Cast the result as the circuit breaker execute() returns Any
            result = await self.circuit_breaker.execute(
                self._fetch_ohlcv_data_from_api, trading_pair, start_time, end_time, interval
            )
            return cast(Optional[pd.DataFrame], result)

        except Exception as e:
            self.logger.error(
                f"Error fetching OHLCV data: {e}", source_module=self._source_module, exc_info=True
            )
            return None

    async def _fetch_ohlcv_data_from_api(
        self, trading_pair: str, start_time: datetime, end_time: datetime, interval: str
    ) -> Optional[pd.DataFrame]:
        """Actual implementation of API call to Kraken."""
        # TODO: Implement actual API call using aiohttp or ccxt
        # This is a placeholder for the actual implementation

        # For testing purposes, generate some dummy data
        self.logger.warning(
            "Using dummy data for testing purposes", source_module=self._source_module
        )

        # Create date range with appropriate interval
        interval_seconds = self._interval_to_seconds(interval)
        date_range = pd.date_range(start=start_time, end=end_time, freq=f"{interval_seconds}S")

        if len(date_range) == 0:
            return None

        # Generate dummy data
        import numpy as np

        # More precise typing for the list of dictionaries
        data_element_type = Dict[str, Union[pd.Timestamp, float, int]]  # numpy can yield int/float
        data: List[data_element_type] = []
        base_price = 100.0

        for timestamp in date_range:  # timestamp is pd.Timestamp
            # Random price movement
            price_change = (np.random.random() - 0.5) * 2  # float
            close_price = base_price * (1 + price_change * 0.01)  # float
            high_price = close_price * (1 + np.random.random() * 0.005)  # float
            low_price = close_price * (1 - np.random.random() * 0.005)  # float
            open_price = close_price * (1 + (np.random.random() - 0.5) * 0.01)  # float
            volume = np.random.random() * 100  # float

            data.append(
                {
                    "timestamp": timestamp,  # pd.Timestamp
                    "open": open_price,  # float
                    "high": high_price,  # float
                    "low": low_price,  # float
                    "close": close_price,  # float
                    "volume": volume,  # float
                }
            )

            # Update base price for next iteration
            base_price = close_price

        # Cast was redundant here, mypy can infer pd.DataFrame(data) is DataFrame
        constructed_df: pd.DataFrame = pd.DataFrame(data)
        constructed_df.set_index("timestamp", inplace=True)

        # Simulate network delay
        await asyncio.sleep(0.5)

        return constructed_df

    def _interval_to_seconds(self, interval: str) -> int:
        """Convert interval string to seconds."""
        unit = interval[-1].lower()
        value = int(interval[:-1])

        if unit == "m":
            return value * 60
        elif unit == "h":
            return value * 60 * 60
        elif unit == "d":
            return value * 24 * 60 * 60
        else:
            self.logger.warning(
                f"Unknown interval unit: {unit}", source_module=self._source_module
            )
            return 60  # Default to 1 minute

    def _validate_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """Validate OHLCV data for correctness and completeness."""
        if df is None or df.empty:
            self.logger.warning(
                "Empty DataFrame provided for validation", source_module=self._source_module
            )
            return False

        # Check required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        for column in required_columns:
            if column not in df.columns:
                self.logger.warning(
                    f"Missing required column: {column}", source_module=self._source_module
                )
                return False

        # Check for NaN values
        if df[required_columns].isna().any().any():
            self.logger.warning(
                "NaN values found in OHLCV data", source_module=self._source_module
            )
            return False

        # Check for negative values
        if (df[required_columns] < 0).any().any():
            self.logger.warning(
                "Negative values found in OHLCV data", source_module=self._source_module
            )
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
                "Invalid OHLC relationship found", source_module=self._source_module
            )
            return False

        return True

    async def _store_ohlcv_data_in_influxdb(
        self, df: pd.DataFrame, trading_pair: str, interval: str
    ) -> bool:
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
                f"Stored {len(points)} OHLCV points in InfluxDB for {trading_pair}",
                source_module=self._source_module,
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Error storing OHLCV data in InfluxDB: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
            return False

    async def _query_ohlcv_data_from_influxdb(
        self, trading_pair: str, start_time: datetime, end_time: datetime, interval: str
    ) -> Optional[pd.DataFrame]:
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
                        }
                    )

            if not records:
                self.logger.debug(
                    f"No OHLCV data found in InfluxDB for {trading_pair}",
                    source_module=self._source_module,
                )
                return None

            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)

            return df

        except Exception as e:
            self.logger.error(
                f"Error querying OHLCV data from InfluxDB: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
            return None

    async def _query_trades_data_from_influxdb(
        self, trading_pair: str, start_time: datetime, end_time: datetime
    ) -> Optional[pd.DataFrame]:
        """Query trade data from InfluxDB."""
        # Similar to _query_ohlcv_data_from_influxdb but for trade data
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
                        }
                    )

            if not records:
                self.logger.debug(
                    f"No trade data found in InfluxDB for {trading_pair}",
                    source_module=self._source_module,
                )
                return None

            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)

            return df

        except Exception as e:
            self.logger.error(
                f"Error querying trade data from InfluxDB: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
            return None

    def _get_missing_ranges(
        self, df: Optional[pd.DataFrame], start_time: datetime, end_time: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Determine what date ranges are missing from the data."""
        if df is None or df.empty:
            # All data is missing
            return [(start_time, end_time)]

        missing_ranges = []

        # Check if data starts after requested start time
        if df.index.min() > start_time:
            missing_ranges.append((start_time, df.index.min()))

        # Check if data ends before requested end time
        if df.index.max() < end_time:
            missing_ranges.append((df.index.max(), end_time))

        # TODO: Check for gaps within the data range
        # This would require sorting the data and checking for expected intervals

        return missing_ranges

    async def _get_latest_timestamp_from_influxdb(
        self, trading_pair: str, interval: str
    ) -> Optional[datetime]:
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
                        f"Unexpected type for record time: {type(record_time)}",
                        source_module=self._source_module,
                    )
                    return None

            return None

        except Exception as e:
            self.logger.error(
                f"Error getting latest timestamp from InfluxDB: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
            return None
