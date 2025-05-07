"""Tests for the KrakenHistoricalDataService implementation."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

from gal_friday.kraken_historical_data_service import (
    KrakenHistoricalDataService,
    RateLimitTracker,
    CircuitBreaker
)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.debug = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    return logger


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return {
        "influxdb": {
            "url": "http://localhost:8086",
            "token": "test_token",
            "org": "test_org",
            "bucket": "test_bucket"
        },
        "api_tier": "default",
        "failure_threshold": 3,
        "reset_timeout": 60,
        "debug": True
    }


@pytest.fixture
def historical_data_service(test_config, mock_logger):
    """Create a historical data service with mocked dependencies."""
    # Mock InfluxDB client
    with patch("gal_friday.kraken_historical_data_service.InfluxDBClient") as mock_client:
        # Setup mock return values
        mock_client.return_value.write_api.return_value = MagicMock()
        mock_client.return_value.query_api.return_value = MagicMock()
        
        # Create service
        service = KrakenHistoricalDataService(test_config, mock_logger)
        
        yield service


class TestRateLimitTracker:
    """Tests for RateLimitTracker class."""

    @pytest.mark.asyncio
    async def test_wait_if_needed(self, mock_logger):
        """Test that wait_if_needed works correctly."""
        # Create tracker with 1 request per second
        tracker = RateLimitTracker(tier="default", logger=mock_logger)
        
        # Make initial request to set last_request_time
        await tracker.wait_if_needed()
        first_request_time = tracker.last_request_time
        
        # Make immediate second request, should wait
        await tracker.wait_if_needed()
        second_request_time = tracker.last_request_time
        
        # Should have waited close to 1 second
        time_diff = (second_request_time - first_request_time).total_seconds()
        assert time_diff >= 0.9, f"Wait time too short: {time_diff}"
        
        # Logger should have been called with wait message
        mock_logger.debug.assert_called()


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self, mock_logger):
        """Test that circuit breaker allows successful calls."""
        circuit = CircuitBreaker(logger=mock_logger)
        
        # Mock successful function
        async def success_func():
            return "success"
            
        result = await circuit.execute(success_func)
        assert result == "success"
        assert circuit.state == "CLOSED"
        assert circuit.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure(self, mock_logger):
        """Test that circuit breaker opens after failures."""
        circuit = CircuitBreaker(failure_threshold=2, logger=mock_logger)
        
        # Mock failing function
        async def fail_func():
            raise Exception("Test failure")
            
        # First failure
        with pytest.raises(Exception):
            await circuit.execute(fail_func)
        assert circuit.state == "CLOSED"
        assert circuit.failure_count == 1
        
        # Second failure should open circuit
        with pytest.raises(Exception):
            await circuit.execute(fail_func)
        assert circuit.state == "OPEN"
        assert circuit.failure_count == 2
        
        # Next call should be blocked without executing function
        with pytest.raises(Exception) as excinfo:
            await circuit.execute(fail_func)
        assert "Circuit breaker is OPEN" in str(excinfo.value)


class TestKrakenHistoricalDataService:
    """Tests for KrakenHistoricalDataService class."""

    def test_initialization(self, historical_data_service, mock_logger):
        """Test that the service initializes correctly."""
        assert historical_data_service.logger == mock_logger
        mock_logger.info.assert_called_with(
            "KrakenHistoricalDataService initialized", 
            source_module="KrakenHistoricalDataService"
        )

    @pytest.mark.asyncio
    async def test_validate_ohlcv_data(self, historical_data_service):
        """Test OHLCV data validation."""
        # Valid data
        valid_data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [95.0, 96.0],
            'close': [102.0, 103.0],
            'volume': [1000.0, 1100.0]
        })
        
        assert historical_data_service._validate_ohlcv_data(valid_data) is True
        
        # Invalid data - missing column
        invalid_data1 = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [95.0, 96.0],
            'close': [102.0, 103.0]
            # Missing volume
        })
        
        assert historical_data_service._validate_ohlcv_data(invalid_data1) is False
        
        # Invalid data - negative value
        invalid_data2 = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [-95.0, 96.0],  # Negative low
            'close': [102.0, 103.0],
            'volume': [1000.0, 1100.0]
        })
        
        assert historical_data_service._validate_ohlcv_data(invalid_data2) is False
        
        # Invalid data - OHLC relationship
        invalid_data3 = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [95.0, 106.0],  # High less than open
            'low': [90.0, 96.0],
            'close': [102.0, 103.0],
            'volume': [1000.0, 1100.0]
        })
        
        assert historical_data_service._validate_ohlcv_data(invalid_data3) is False

    @pytest.mark.asyncio
    async def test_get_missing_ranges(self, historical_data_service):
        """Test missing ranges calculation."""
        # Create a test dataframe with time index
        start = datetime(2023, 1, 1, 0, 0, 0)
        end = datetime(2023, 1, 1, 5, 0, 0)
        
        # Data covers 1:00 to 3:00, missing 0:00-1:00 and 3:00-5:00
        data_times = [
            datetime(2023, 1, 1, 1, 0, 0),
            datetime(2023, 1, 1, 2, 0, 0),
            datetime(2023, 1, 1, 3, 0, 0)
        ]
        
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, 97.0],
            'close': [102.0, 103.0, 104.0],
            'volume': [1000.0, 1100.0, 1200.0]
        }, index=data_times)
        
        # Test with data partially covering range
        missing = historical_data_service._get_missing_ranges(df, start, end)
        assert len(missing) == 2
        assert missing[0][0] == start
        assert missing[0][1] == datetime(2023, 1, 1, 1, 0, 0)
        assert missing[1][0] == datetime(2023, 1, 1, 3, 0, 0)
        assert missing[1][1] == end
        
        # Test with empty dataframe
        missing = historical_data_service._get_missing_ranges(None, start, end)
        assert len(missing) == 1
        assert missing[0][0] == start
        assert missing[0][1] == end

    @pytest.mark.asyncio
    async def test_interval_to_seconds(self, historical_data_service):
        """Test interval string conversion to seconds."""
        assert historical_data_service._interval_to_seconds("1m") == 60
        assert historical_data_service._interval_to_seconds("5m") == 300
        assert historical_data_service._interval_to_seconds("1h") == 3600
        assert historical_data_service._interval_to_seconds("1d") == 86400
        assert historical_data_service._interval_to_seconds("1x") == 60  # Unknown unit defaults to 1m

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_data_from_api(self, historical_data_service):
        """Test fetching OHLCV data from API (dummy implementation)."""
        start = datetime(2023, 1, 1, 0, 0, 0)
        end = datetime(2023, 1, 1, 1, 0, 0)
        
        result = await historical_data_service._fetch_ohlcv_data_from_api(
            "BTC/USD", start, end, "1m"
        )
        
        # Should return a dataframe with 61 rows (0:00 to 1:00 inclusive)
        assert result is not None
        assert len(result) == 61
        assert result.index[0] == start
        assert result.index[-1] == end
        
        # Should have all required columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in result.columns 