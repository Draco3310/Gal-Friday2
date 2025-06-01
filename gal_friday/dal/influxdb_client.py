"""InfluxDB client for time-series data."""

from datetime import datetime
from typing import Any

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.write_api import SYNCHRONOUS

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService


class TimeSeriesDB:
    """InfluxDB client wrapper for time-series data operations."""

    def __init__(self, config: ConfigManager, logger: LoggerService) -> None:
        """Initialize the InfluxDB client.

        Args:
            config: Configuration manager instance
            logger: Logger service instance
        """
        self.config = config
        self.logger = logger
        self._client: InfluxDBClient | None = None
        self._source_module = self.__class__.__name__

        # Initialize InfluxDB client
        self._client = InfluxDBClient(
            url=config.get("influxdb.url"),
            token=config.get("influxdb.token"),
            org=config.get("influxdb.org"),
        )

        self.write_api = self._client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self._client.query_api()
        self.bucket = config.get("influxdb.bucket", "market-data")

    async def write_points(self, points: list[Point]) -> bool:
        """Write points to InfluxDB.

        Args:
            points: List of point dictionaries to write

        Returns:
            bool: True if write was successful, False otherwise

        Raises:
            RuntimeError: If client is not initialized
            InfluxDBError: If there's an error writing points
        """
        if not self._client:
            raise RuntimeError("InfluxDB client is not initialized")

        try:
            for point in points:
                self.write_api.write(bucket=self.bucket, record=point)
            return True
        except InfluxDBError as e:
            self.logger.exception(
                "Error writing points to InfluxDB",
                source_module=self._source_module,
            )
            return False

    async def write_market_data(
        self,
        trading_pair: str,
        exchange: str,
        data_type: str,  # 'ohlcv', 'tick', 'orderbook'
        data: dict[str, Any],
    ) -> None:
        """Write market data point."""
        try:
            point = Point("market_data") \
                .tag("trading_pair", trading_pair) \
                .tag("exchange", exchange) \
                .tag("data_type", data_type)

            # Add fields based on data type
            if data_type == "ohlcv":
                point.field("open", float(data["open"])) \
                     .field("high", float(data["high"])) \
                     .field("low", float(data["low"])) \
                     .field("close", float(data["close"])) \
                     .field("volume", float(data["volume"]))

            elif data_type == "tick":
                point.field("price", float(data["price"])) \
                     .field("volume", float(data["volume"])) \
                     .field("side", data["side"])

            elif data_type == "orderbook":
                point.field("bid", float(data["bid"])) \
                     .field("ask", float(data["ask"])) \
                     .field("bid_volume", float(data["bid_volume"])) \
                     .field("ask_volume", float(data["ask_volume"])) \
                     .field("spread", float(data["spread"]))

            # Set timestamp
            if "timestamp" in data:
                point.time(data["timestamp"])

            await self.write_points([point])

        except Exception:
            self.logger.exception(
                "Error writing market data to InfluxDB",
                source_module=self._source_module,
            )

    async def write_metrics(
        self,
        metric_name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Write system metrics."""
        try:
            point = Point("system_metrics") \
                .field(metric_name, value)

            if tags:
                for key, val in tags.items():
                    point.tag(key, val)

            await self.write_points([point])

        except Exception:
            self.logger.exception(
                "Error writing metrics to InfluxDB",
                source_module=self._source_module,
            )

    async def query(self, query: str) -> list[dict[str, Any]]:
        """Query data from InfluxDB.

        Args:
            query: InfluxDB query string

        Returns:
            List of result dictionaries

        Raises:
            RuntimeError: If client is not initialized
            InfluxDBError: If there's an error executing the query
        """
        if not self._client:
            raise RuntimeError("InfluxDB client is not initialized")

        try:
            result = self.query_api.query(query=query)

            results = []
            for table in result:
                for record in table.records:
                    results.append({
                        "timestamp": record.get_time(),
                        "values": record.values,
                    })

            return results

        except InfluxDBError as e:
            self.logger.exception(
                "Error querying data from InfluxDB",
                source_module=self._source_module,
            )
            return []

    async def query_ohlcv(self,
                         trading_pair: str,
                         timeframe: str,
                         start_time: datetime,
                         end_time: datetime) -> list[dict[str, Any]]:
        """Query OHLCV data for a time range."""
        query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> filter(fn: (r) => r["_measurement"] == "market_data")
            |> filter(fn: (r) => r["trading_pair"] == "{trading_pair}")
            |> filter(fn: (r) => r["data_type"] == "ohlcv")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> window(every: {timeframe})
            |> sort(columns: ["_time"])
        """

        try:
            result = self.query_api.query(query=query)

            candles = []
            for table in result:
                for record in table.records:
                    candles.append({
                        "timestamp": record.get_time(),
                        "open": record.values.get("open"),
                        "high": record.values.get("high"),
                        "low": record.values.get("low"),
                        "close": record.values.get("close"),
                        "volume": record.values.get("volume"),
                    })

            return candles

        except Exception:
            self.logger.exception(
                "Error querying OHLCV data from InfluxDB",
                source_module=self._source_module,
            )
            return []

    def close(self) -> None:
        """Close InfluxDB client."""
        if self._client:
            self._client.close()
