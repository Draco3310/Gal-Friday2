"""Unit tests for the Kraken market price service implementation."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService
from gal_friday.market_price.kraken_service import KrakenMarketPriceService


class TestKrakenMarketPriceService:
    """Tests for the KrakenMarketPriceService class."""

    @pytest.fixture
    def config_manager(self):
        """Create a mock config manager."""
        config = MagicMock(spec=ConfigManager)

        # Mock required config methods
        config.get.return_value = "https://api.kraken.com"
        config.get_int.return_value = 3
        config.get_float.return_value = 1.0
        config.get_dict.return_value = {"BTC": "XBT", "DOGE": "XDG"}
        config.get_list.return_value = ["USD", "EUR", "BTC"]

        return config

    @pytest.fixture
    def logger_service(self):
        """Create a mock logger service."""
        logger = MagicMock(spec=LoggerService)
        return logger

    @pytest.fixture
    async def kraken_service(self, config_manager, logger_service):
        """Create a KrakenMarketPriceService instance."""
        service = KrakenMarketPriceService(config_manager, logger_service)
        await service.start()
        yield service
        await service.stop()

    @pytest.mark.asyncio
    async def test_session_validation_decorator(self, config_manager, logger_service):
        """Test that the session validation decorator raises ValueError.

        when session is not initialized.
        """
        service = KrakenMarketPriceService(config_manager, logger_service)

        # Attempt to call method without starting the service (creating the session)
        with pytest.raises(ValueError, match="Session not initialized"):
            await service.get_latest_price("BTC/USD")

    @pytest.mark.asyncio
    async def test_session_validation_allows_request_after_start(self, kraken_service):
        """Test that the session validation decorator allows requests after session is created."""
        # Mock the _make_api_request method to avoid actual HTTP requests
        with patch.object(kraken_service, "_make_api_request", new=AsyncMock()) as mock_request:
            # Set up the mock to return a valid response
            mock_request.return_value = {"result": {"XXBTZUSD": {"c": ["50000.0", "1.0"]}}}

            # This should not raise ValueError since we called start() in the fixture
            result = await kraken_service.get_latest_price("BTC/USD")

            # Verify _make_api_request was called
            mock_request.assert_called_once()

            # Verify the result was parsed correctly
            assert result == Decimal("50000.0")

    @pytest.mark.asyncio
    async def test_pair_mapping(self, kraken_service):
        """Test the internal to Kraken pair mapping logic."""
        # Test standard mapping
        assert kraken_service._map_internal_to_kraken_pair("BTC/USD") == "XXBTZUSD"

        # Test with alternative names
        assert kraken_service._map_internal_to_kraken_pair("DOGE/EUR") == "XDGZEUR"

        # Test with invalid format
        assert kraken_service._map_internal_to_kraken_pair("INVALID") is None

        # Test with empty input
        assert kraken_service._map_internal_to_kraken_pair("") is None

    @pytest.mark.asyncio
    async def test_convert_amount_same_currency(self, kraken_service):
        """Test converting between the same currency returns the original amount."""
        result = await kraken_service.convert_amount(Decimal("10.0"), "BTC", "BTC")
        assert result == Decimal("10.0")

    @pytest.mark.asyncio
    async def test_convert_amount_direct_conversion(self, kraken_service):
        """Test direct currency conversion."""
        # Mock _get_safe_price to return a simulated price for BTC/USD
        with patch.object(kraken_service, "_get_safe_price", new=AsyncMock()) as mock_get_price:
            mock_get_price.side_effect = lambda pair: {"BTC/USD": Decimal("50000.0")}.get(pair)

            # Convert 2 BTC to USD
            result = await kraken_service.convert_amount(Decimal("2.0"), "BTC", "USD")

            # Verify conversion used the right price
            assert result == Decimal("100000.0")  # 2 BTC * $50,000

            # Verify _get_safe_price was called with the right pair
            mock_get_price.assert_called_with("BTC/USD")

    @pytest.mark.asyncio
    async def test_convert_amount_reverse_conversion(self, kraken_service):
        """Test reverse currency conversion."""
        # Mock _get_safe_price to return None for direct path, but a price for reverse path
        with patch.object(kraken_service, "_get_safe_price", new=AsyncMock()) as mock_get_price:
            mock_get_price.side_effect = lambda pair: {
                "BTC/EUR": None,  # No direct path
                "EUR/BTC": Decimal("0.00002"),  # 1 EUR = 0.00002 BTC (50,000 EUR/BTC)
            }.get(pair)

            # Convert 2 BTC to EUR
            result = await kraken_service.convert_amount(Decimal("2.0"), "BTC", "EUR")

            # Verify conversion used the reverse path correctly
            assert result == Decimal("100000.0")  # 2 BTC / 0.00002 BTC/EUR = 100,000 EUR

    @pytest.mark.asyncio
    async def test_convert_amount_via_intermediary(self, kraken_service):
        """Test conversion through an intermediary currency."""
        # Mock _get_safe_price to simulate conversion via USD
        with patch.object(kraken_service, "_get_safe_price", new=AsyncMock()) as mock_get_price:
            mock_get_price.side_effect = lambda pair: {
                "ETH/USD": Decimal("3000.0"),  # 1 ETH = $3,000
                "USD/JPY": Decimal("110.0"),  # 1 USD = 110 JPY
            }.get(pair)

            # Convert 5 ETH to JPY
            result = await kraken_service.convert_amount(Decimal("5.0"), "ETH", "JPY")

            # Verify the conversion went through both steps correctly
            # 5 ETH * $3,000 = $15,000
            # $15,000 * 110 JPY = 1,650,000 JPY
            assert result == Decimal("1650000.0")

    @pytest.mark.asyncio
    async def test_convert_amount_failed_conversion(self, kraken_service):
        """Test handling of failed conversions."""
        # Mock _get_safe_price to always return None, simulating no conversion path
        with patch.object(kraken_service, "_get_safe_price", new=AsyncMock(return_value=None)):
            # Attempt to convert a currency with no path
            result = await kraken_service.convert_amount(Decimal("10.0"), "XYZ", "ABC")

            # Verify result is None, indicating conversion failed
            assert result is None

            # Verify logger received a warning
            kraken_service.logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_retry_logic_on_api_errors(self, kraken_service):
        """Test the retry logic for API errors."""
        # Create a mock that fails twice with a rate limit error, then succeeds
        response_side_effect = [
            {"error": ["Rate limit exceeded"]},
            {"error": ["Rate limit exceeded"]},
            {"result": {"XXBTZUSD": {"c": ["50000.0", "1.0"]}}},
        ]

        with patch.object(kraken_service, "_session") as mock_session:
            # Mock the context manager for aiohttp response
            mock_cm = MagicMock()
            mock_response = AsyncMock()

            # Set up the mock response to always return status 200
            mock_response.status = 200

            # Set up the JSON method to return our sequence of responses
            mock_response.json = AsyncMock(side_effect=response_side_effect)

            # Set up the context manager to return our mock response
            mock_cm.__aenter__.return_value = mock_response
            mock_session.get.return_value = mock_cm

            # Patch _wait_before_retry to avoid actual delays in the test
            with patch.object(kraken_service, "_wait_before_retry", new=AsyncMock()):
                # This should retry twice and then succeed
                result = await kraken_service.get_latest_price("BTC/USD")

                # Verify we got the correct final result
                assert result == Decimal("50000.0")

                # Verify _wait_before_retry was called twice (for the first two failures)
                assert kraken_service._wait_before_retry.call_count == 2
