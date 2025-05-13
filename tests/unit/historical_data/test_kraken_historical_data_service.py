"""
Tests for the KrakenHistoricalDataService implementation.

This test suite verifies that the KrakenHistoricalDataService correctly implements
the HistoricalDataService interface and properly handles error conditions.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from gal_friday.interfaces.historical_data_service_interface import HistoricalDataService
from gal_friday.kraken_historical_data_service import (
    CircuitBreaker,
    KrakenHistoricalDataService,
    RateLimitTracker,
)
from gal_friday.logger_service import LoggerService


class TestKrakenHistoricalDataService:
    """Tests for the KrakenHistoricalDataService implementation."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger service."""
        logger = MagicMock(spec=LoggerService)
        return logger

    @pytest.fixture
    def mock_influxdb_client(self):
        """Create a mock InfluxDBClient."""
        client = MagicMock()

        # Mock InfluxDB APIs
        client.write_api.return_value = MagicMock()
        client.query_api.return_value = MagicMock()

        return client

    @pytest.fixture
    def mock_config(self):
        """Create a sample configuration."""
        return {
            "influxdb": {
                "url": "http://localhost:8086",
                "token": "test_token",
                "org": "test_org",
                "bucket": "test_bucket",
            },
            "api_tier": "default",
            "failure_threshold": 3,
            "reset_timeout": 60,
        }

    @pytest.fixture
    def historical_service(self, mock_config, mock_logger):
        """Create a KrakenHistoricalDataService instance with mocked components."""
        with patch("influxdb_client.InfluxDBClient") as mock_influx:
            mock_influx.return_value = MagicMock()
            mock_influx.return_value.write_api.return_value = MagicMock()
            mock_influx.return_value.query_api.return_value = MagicMock()

            service = KrakenHistoricalDataService(mock_config, mock_logger)
            yield service

    def test_interface_compliance(self, historical_service):
        """Test that KrakenHistoricalDataService implements the HistoricalDataService interface."""
        assert isinstance(historical_service, HistoricalDataService)
        assert hasattr(historical_service, "get_historical_ohlcv")
        assert hasattr(historical_service, "get_historical_trades")
        assert hasattr(historical_service, "get_next_bar")
        assert hasattr(historical_service, "get_atr")

    @pytest.mark.asyncio
    async def test_rate_limiter(self, mock_logger):
        """Test the rate limiter behavior."""
        rate_limiter = RateLimitTracker(tier="default", logger=mock_logger)

        # Measure time to execute two requests
        start_time = datetime.now()

        await rate_limiter.wait_if_needed()  # First request should not wait
        await rate_limiter.wait_if_needed()  # Second request should wait

        elapsed = (datetime.now() - start_time).total_seconds()

        # For the default tier (1 req/sec), the second request should wait close to 1 second
        assert elapsed >= 0.9, "Rate limiter should enforce delay between requests"

        # Test other tiers
        rate_limiter = RateLimitTracker(tier="pro", logger=mock_logger)
        start_time = datetime.now()

        for _ in range(6):  # Pro tier allows 5 req/sec
            await rate_limiter.wait_if_needed()

        elapsed = (datetime.now() - start_time).total_seconds()
        assert elapsed >= 0.9, "Rate limiter should enforce tier-specific limits"

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, mock_logger):
        """Test the circuit breaker behavior."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=2, reset_timeout=0.5, logger=mock_logger  # Short timeout for testing
        )

        # Mock function that succeeds
        async def success_func():
            return "success"

        # Mock function that fails
        async def failing_func():
            raise Exception("Test failure")

        # Test successful execution
        result = await circuit_breaker.execute(success_func)
        assert result == "success"
        assert circuit_breaker.state == "CLOSED"

        # Test handling of failures
        with pytest.raises(Exception):
            await circuit_breaker.execute(failing_func)
        assert circuit_breaker.failure_count == 1
        assert circuit_breaker.state == "CLOSED"  # Still closed after 1 failure

        # Second failure should open the circuit
        with pytest.raises(Exception):
            await circuit_breaker.execute(failing_func)
        assert circuit_breaker.failure_count == 2
        assert circuit_breaker.state == "OPEN"

        # Circuit is open, should reject any request
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await circuit_breaker.execute(success_func)

        # Wait for circuit to transition to HALF-OPEN
        await asyncio.sleep(0.6)  # Slightly longer than reset_timeout

        # In HALF-OPEN state, request should be attempted
        result = await circuit_breaker.execute(success_func)
        assert result == "success"
        assert circuit_breaker.state == "CLOSED"  # Success in HALF-OPEN transitions to CLOSED

    @pytest.mark.asyncio
    async def test_get_historical_ohlcv(self, historical_service):
        """Test the get_historical_ohlcv method with mocked data flow."""
        # Mock _query_ohlcv_data_from_influxdb to return None (no data in InfluxDB)
        historical_service._query_ohlcv_data_from_influxdb = AsyncMock(return_value=None)

        # Mock _fetch_ohlcv_data to return sample data
        sample_data = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            },
            index=pd.date_range(start=datetime.now(timezone.utc), periods=2, freq="1H"),
        )

        historical_service._fetch_ohlcv_data = AsyncMock(return_value=sample_data)

        # Mock _store_ohlcv_data_in_influxdb
        historical_service._store_ohlcv_data_in_influxdb = AsyncMock(return_value=True)

        # Test the method
        start_time = datetime.now(timezone.utc) - timedelta(hours=2)
        end_time = datetime.now(timezone.utc)
        result = await historical_service.get_historical_ohlcv(
            "BTC/USD", start_time, end_time, "1h"
        )

        # Verify results
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in ["open", "high", "low", "close", "volume"])

        # Verify method calls
        historical_service._query_ohlcv_data_from_influxdb.assert_called_once()
        historical_service._fetch_ohlcv_data.assert_called_once()
        historical_service._store_ohlcv_data_in_influxdb.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_historical_ohlcv_with_influxdb_data(self, historical_service):
        """Test get_historical_ohlcv when complete data exists in InfluxDB."""
        # Create sample data that covers the entire requested range
        start_time = datetime.now(timezone.utc) - timedelta(hours=2)
        end_time = datetime.now(timezone.utc)

        sample_data = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000.0, 1100.0, 1200.0],
            },
            index=pd.date_range(
                start=start_time - timedelta(minutes=10),
                end=end_time + timedelta(minutes=10),
                periods=3,
            ),
        )

        # Mock _query_ohlcv_data_from_influxdb to return sample data
        historical_service._query_ohlcv_data_from_influxdb = AsyncMock(return_value=sample_data)

        # Mock _fetch_ohlcv_data to verify it's not called
        historical_service._fetch_ohlcv_data = AsyncMock()

        # Test the method
        result = await historical_service.get_historical_ohlcv(
            "BTC/USD", start_time, end_time, "1h"
        )

        # Verify results
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

        # Verify method calls
        historical_service._query_ohlcv_data_from_influxdb.assert_called_once()
        historical_service._fetch_ohlcv_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_historical_ohlcv_with_partial_data(self, historical_service):
        """Test get_historical_ohlcv when partial data exists in InfluxDB."""
        # Create sample data that only covers part of the requested range
        start_time = datetime.now(timezone.utc) - timedelta(hours=2)
        end_time = datetime.now(timezone.utc)

        # Data only covers the first hour
        partial_data = pd.DataFrame(
            {
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000.0],
            },
            index=pd.date_range(start=start_time, periods=1, freq="1H"),
        )

        # Mock _query_ohlcv_data_from_influxdb to return partial data
        historical_service._query_ohlcv_data_from_influxdb = AsyncMock(return_value=partial_data)

        # New data to be fetched from API for the second hour
        new_data = pd.DataFrame(
            {
                "open": [101.0],
                "high": [103.0],
                "low": [100.0],
                "close": [102.0],
                "volume": [1100.0],
            },
            index=pd.date_range(start=start_time + timedelta(hours=1), periods=1, freq="1H"),
        )

        # Mock _fetch_ohlcv_data to return the new data
        historical_service._fetch_ohlcv_data = AsyncMock(return_value=new_data)

        # Mock _store_ohlcv_data_in_influxdb
        historical_service._store_ohlcv_data_in_influxdb = AsyncMock(return_value=True)

        # Mock _get_missing_ranges to correctly identify missing range
        historical_service._get_missing_ranges = MagicMock(
            return_value=[(start_time + timedelta(hours=1), end_time)]
        )

        # Test the method
        result = await historical_service.get_historical_ohlcv(
            "BTC/USD", start_time, end_time, "1h"
        )

        # Verify results
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == 2  # Should have data for both hours

        # Verify method calls
        historical_service._query_ohlcv_data_from_influxdb.assert_called_once()
        historical_service._fetch_ohlcv_data.assert_called_once()
        historical_service._store_ohlcv_data_in_influxdb.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_historical_trades(self, historical_service):
        """Test the get_historical_trades method."""
        # Mock _query_trades_data_from_influxdb to return sample data
        sample_trades = pd.DataFrame(
            {
                "price": [100.0, 101.0, 102.0],
                "volume": [1.0, 2.0, 0.5],
                "side": ["buy", "sell", "buy"],
            },
            index=pd.date_range(
                start=datetime.now(timezone.utc) - timedelta(hours=1), periods=3, freq="20min"
            ),
        )

        historical_service._query_trades_data_from_influxdb = AsyncMock(return_value=sample_trades)

        # Test the method
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)
        result = await historical_service.get_historical_trades("BTC/USD", start_time, end_time)

        # Verify results
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in ["price", "volume", "side"])

        # Verify method calls
        historical_service._query_trades_data_from_influxdb.assert_called_once()

    def test_get_next_bar(self, historical_service):
        """Test the get_next_bar synchronous method."""
        # Mock query_api.query to return a sample bar
        tables = MagicMock()
        record = MagicMock()
        record.values = {
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 1000.0,
            "_time": datetime.now(timezone.utc),
        }
        tables.records = [record]

        historical_service.query_api.query = MagicMock(return_value=[tables])

        # Test the method
        result = historical_service.get_next_bar(
            "BTC/USD", datetime.now(timezone.utc) - timedelta(hours=1)
        )

        # Verify results
        assert result is not None
        assert isinstance(result, pd.Series)
        assert all(key in result.index for key in ["open", "high", "low", "close", "volume"])

        # Test error handling
        historical_service.query_api.query = MagicMock(side_effect=Exception("Test exception"))
        result = historical_service.get_next_bar(
            "BTC/USD", datetime.now(timezone.utc) - timedelta(hours=1)
        )
        assert result is None
        assert historical_service.logger.error.called

    def test_get_atr(self, historical_service):
        """Test the get_atr synchronous method."""
        # Mock query_api.query to return sample OHLC data
        tables = MagicMock()
        records = []

        # Create 15 sample records for ATR calculation (need at least 14 by default)
        start_time = datetime.now(timezone.utc) - timedelta(days=15)
        for i in range(15):
            record = MagicMock()
            record.values = {
                "high": 100.0 + i,
                "low": 95.0 + i,
                "close": 98.0 + i,
                "_time": start_time + timedelta(days=i),
            }
            records.append(record)

        tables.records = records
        historical_service.query_api.query = MagicMock(return_value=[tables])

        # Test the method
        result = historical_service.get_atr("BTC/USD", datetime.now(timezone.utc), period=14)

        # Verify results
        assert result is not None
        assert isinstance(result, Decimal)
        assert result > Decimal("0")  # ATR should be positive

        # Test with empty data
        historical_service.query_api.query = MagicMock(return_value=[])
        result = historical_service.get_atr("BTC/USD", datetime.now(timezone.utc), period=14)
        assert result is None

        # Test error handling
        historical_service.query_api.query = MagicMock(side_effect=Exception("Test exception"))
        result = historical_service.get_atr("BTC/USD", datetime.now(timezone.utc), period=14)
        assert result is None
        assert historical_service.logger.error.called

    def test_interval_to_seconds(self, historical_service):
        """Test the _interval_to_seconds method."""
        assert historical_service._interval_to_seconds("1m") == 60
        assert historical_service._interval_to_seconds("5m") == 300
        assert historical_service._interval_to_seconds("1h") == 3600
        assert historical_service._interval_to_seconds("1d") == 86400

        # Test unknown unit (should default to 60 seconds)
        assert historical_service._interval_to_seconds("1x") == 60
        assert historical_service.logger.warning.called

    def test_validate_ohlcv_data(self, historical_service):
        """Test the _validate_ohlcv_data method."""
        # Valid data
        valid_data = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            }
        )
        assert historical_service._validate_ohlcv_data(valid_data) is True

        # Missing column
        invalid_data = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                # Missing 'close'
                "volume": [1000.0, 1100.0],
            }
        )
        assert historical_service._validate_ohlcv_data(invalid_data) is False

        # NaN values
        invalid_data = pd.DataFrame(
            {
                "open": [100.0, np.nan],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            }
        )
        assert historical_service._validate_ohlcv_data(invalid_data) is False

        # Negative values
        invalid_data = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, -1.0],  # Negative low
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            }
        )
        assert historical_service._validate_ohlcv_data(invalid_data) is False

        # Invalid OHLC relationship (high < low)
        invalid_data = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 99.0],  # High < low
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            }
        )
        assert historical_service._validate_ohlcv_data(invalid_data) is False

        # Empty DataFrame
        assert historical_service._validate_ohlcv_data(pd.DataFrame()) is False

        # None
        assert historical_service._validate_ohlcv_data(None) is False

    @pytest.mark.asyncio
    async def test_get_historical_ohlcv_with_from_param(self, historical_service):
        """Test get_historical_ohlcv when from parameter is used."""
        # Mock _query_ohlcv_data_from_influxdb to return None (no data in InfluxDB)
        historical_service._query_ohlcv_data_from_influxdb = AsyncMock(return_value=None)

        # Mock _fetch_ohlcv_data to return sample data
        sample_data = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            },
            index=pd.date_range(start=datetime.now(timezone.utc), periods=2, freq="1H"),
        )

        historical_service._fetch_ohlcv_data = AsyncMock(return_value=sample_data)

        # Mock _store_ohlcv_data_in_influxdb
        historical_service._store_ohlcv_data_in_influxdb = AsyncMock(return_value=True)

        # Test the method
        start_time = datetime.now(timezone.utc) - timedelta(hours=2)
        end_time = datetime.now(timezone.utc)
        result = await historical_service.get_historical_ohlcv(
            "BTC/USD", start_time, end_time, "1h", from_param=1672531261
        )

        # Verify results
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in ["open", "high", "low", "close", "volume"])

        # Verify method calls
        historical_service._query_ohlcv_data_from_influxdb.assert_called_once()
        historical_service._fetch_ohlcv_data.assert_called_once()
        historical_service._store_ohlcv_data_in_influxdb.assert_called_once()

        # The service should have attempted to download more data
        # starting from the last known timestamp
        # Using param=from value of 1672531261 (which is the timestamp of the last row + 1)
        assert (
            "from=1672531261" in historical_service._query_ohlcv_data_from_influxdb.call_args[0][0]
        ), "Second request should start from last timestamp"
