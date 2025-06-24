from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
import logging
import time
from typing import Any, cast

import aiohttp
import asyncio

from gal_friday.simulated_market_price_service import DataRequest, HistoricalDataPoint, HistoricalDataProvider
from gal_friday.utils.kraken_api import generate_kraken_signature, prepare_kraken_request_data


class APIError(Exception):
    """Custom exception for API-related errors."""


class RateLimiter:
    """Token bucket rate limiter implementation."""

    def __init__(self, rate: float = 1.0, burst: int = 5) -> None:
        """Initialize the instance."""
        self.rate = rate  # tokens per second
        self.burst = burst  # maximum burst size
        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens from the bucket, waiting if necessary."""
        async with self._lock:
            while self.tokens < tokens:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens < tokens:
                    sleep_time = (tokens - self.tokens) / self.rate
                    await asyncio.sleep(sleep_time)

            self.tokens -= tokens


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        """Initialize the instance."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == "open":
                if self.last_failure_time is not None and time.monotonic() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                else:
                    raise APIError("Circuit breaker is open")

            try:
                result = await func(*args, **kwargs)
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
            except Exception:
                self.failure_count += 1
                self.last_failure_time = time.monotonic()
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                raise
            else:
                return result

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "open"

    def record_failure(self) -> None:
        """Record a failure."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class APIDataProvider(HistoricalDataProvider):
    """Production-ready provider for fetching historical data from Kraken API."""

    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        """Initialize the instance."""
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__

        # API configuration
        self.base_url = config.get("api", {}).get("base_url", "https://api.kraken.com")
        self.api_key = config.get("api", {}).get("key", "")
        self.api_secret = config.get("api", {}).get("secret", "")

        # HTTP client
        self._session: aiohttp.ClientSession | None = None

        # Rate limiting
        rate_limit = config.get("api", {}).get("rate_limit", 1)
        burst_limit = config.get("api", {}).get("burst_limit", 5)
        self._rate_limiter = RateLimiter(rate=rate_limit, burst=burst_limit)

        # Circuit breaker for fault tolerance
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
        )

        # Cache for symbol validation
        self._valid_symbols_cache: dict[str, dict[str, Any]] = {}
        self._cache_ttl = 3600  # 1 hour

        # Query result cache
        self._query_cache: dict[str, dict[str, Any]] = {}
        self._cache_max_size = 50

        # Symbol mapping (internal to Kraken format)
        self._symbol_mapping = {
            "XRP/USD": "XRPUSD",
            "BTC/USD": "XBTUSD",
            "ETH/USD": "ETHUSD",
            # Add more mappings as needed
        }

        # Performance metrics
        self._api_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "average_response_time": 0.0,
        }

    async def initialize(self) -> None:
        """Initialize HTTP session."""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)

            self.logger.info(
                "APIDataProvider initialized",
                extra={"source_module": self._source_module},
            )

    async def cleanup(self) -> None:
        """Clean up HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def fetch_data(self, request: DataRequest) -> list[HistoricalDataPoint]:
        """Fetch historical data from Kraken API."""
        # Ensure initialized
        if not self._session:
            await self.initialize()

        # Validate request
        if not await self.validate_symbol(request.symbol):
            raise ValueError(f"Unsupported symbol: {request.symbol}")

        # Check cache first
        cache_key = self._generate_cache_key(request)
        cached_data = self._query_cache.get(cache_key)
        if cached_data and self._is_cache_valid(cached_data):
            self._api_metrics["cache_hits"] += 1
            return cast("list[HistoricalDataPoint]", cached_data["data"])

        # Rate limit check
        await self._rate_limiter.acquire()

        # Start timer
        start_time = time.monotonic()

        try:
            # Use circuit breaker for API call
            data_points = await self._circuit_breaker.call(
                self._fetch_ohlc_data,
                request,
            )

            # Cache the results
            self._cache_query_result(cache_key, data_points)

            # Record metrics
            response_time = time.monotonic() - start_time
            self._update_api_metrics(True, response_time)

            self.logger.debug(
                f"Fetched {len(data_points)} data points in {response_time:.3f}s",
                extra={
                    "source_module": self._source_module,
                    "symbol": request.symbol,
                    "start_date": request.start_date.isoformat(),
                    "end_date": request.end_date.isoformat(),
                },
            )

            return cast("list[HistoricalDataPoint]", data_points)

        except Exception as e:
            self._circuit_breaker.record_failure()
            self._update_api_metrics(False, time.monotonic() - start_time)

            self.logger.error(
                f"Failed to fetch data from API: {e}",
                extra={
                    "source_module": self._source_module,
                    "symbol": request.symbol,
                },
                exc_info=True,
            )
            raise

    async def _fetch_ohlc_data(self, request: DataRequest) -> list[HistoricalDataPoint]:
        """Fetch OHLC data from Kraken API."""
        # Convert symbol to Kraken format
        kraken_pair = self._map_to_kraken_pair(request.symbol)

        # Convert frequency to Kraken interval
        interval = self._convert_to_kraken_interval(request.frequency)

        # Prepare parameters
        params = {
            "pair": kraken_pair,
            "interval": interval,
        }

        # Add since parameter if start date is provided
        if request.start_date:
            # Kraken expects Unix timestamp
            params["since"] = int(request.start_date.timestamp())

        # Make API request
        response_data = await self._make_request(
            endpoint="/0/public/OHLC",
            params=params,
        )

        # Parse response
        return self._parse_kraken_ohlc_response(response_data, request)

    async def _make_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        retries: int = 3,
    ) -> dict[str, Any]:
        """Make HTTP request to Kraken API with retry logic."""
        if not self._session:
            raise RuntimeError("HTTP session not initialized")

        last_error = None

        for attempt in range(retries):
            try:
                url = f"{self.base_url}{endpoint}"

                # Determine if this is a private endpoint
                is_private = "/private/" in endpoint

                if is_private and data:
                    # Add authentication for private endpoints
                    nonce = int(time.time() * 1000)
                    request_data = prepare_kraken_request_data(data, nonce)

                    headers = {
                        "API-Key": self.api_key,
                        "API-Sign": generate_kraken_signature(
                            endpoint, request_data, nonce, self.api_secret,
                        ),
                    }

                    async with self._session.post(
                        url,
                        data=request_data,
                        headers=headers,
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                else:
                    # Public endpoint
                    async with self._session.get(url, params=params) as response:
                        response.raise_for_status()
                        result = await response.json()

                # Check for API errors
                if result.get("error"):
                    error_msg = ", ".join(result["error"])
                    raise APIError(f"Kraken API error: {error_msg}")

                return cast("dict[str, Any]", result.get("result", {}))

            except aiohttp.ClientError as e:
                last_error = e
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    await asyncio.sleep(wait_time)
                continue

        raise APIError(f"Failed after {retries} attempts: {last_error}")

    def _parse_kraken_ohlc_response(
        self,
        response: dict[str, Any],
        request: DataRequest,
    ) -> list[HistoricalDataPoint]:
        """Parse Kraken OHLC response into HistoricalDataPoint objects."""
        data_points: list[HistoricalDataPoint] = []

        # Find the pair data in response
        self._map_to_kraken_pair(request.symbol)
        pair_data = None

        # Kraken returns data with various pair formats, try to find matching one
        for key, value in response.items():
            if key != "last" and isinstance(value, list):
                pair_data = value
                break

        if not pair_data:
            return data_points

        for candle in pair_data:
            # Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
            if len(candle) < 8:
                continue

            timestamp = datetime.fromtimestamp(candle[0], tz=UTC)

            # Skip if outside requested range
            if request.start_date and timestamp < request.start_date:
                continue
            if request.end_date and timestamp > request.end_date:
                continue

            data_point = HistoricalDataPoint(
                timestamp=timestamp,
                symbol=request.symbol,
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                volume=float(candle[6]),
                metadata={
                    "vwap": float(candle[5]),
                    "trade_count": int(candle[7]),
                    "source": "kraken",
                },
            )

            # Validate data point
            if self._validate_data_point(data_point):
                data_points.append(data_point)
            else:
                self.logger.warning(
                    f"Invalid data point skipped: {data_point}",
                    extra={"source_module": self._source_module},
                )

        return data_points

    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported by checking against Kraken's asset pairs."""
        # Check cache first
        if symbol in self._valid_symbols_cache:
            cache_entry = self._valid_symbols_cache[symbol]
            if self._is_cache_entry_valid(cache_entry):
                return cast("bool", cache_entry["valid"])

        # Ensure initialized
        if not self._session:
            await self.initialize()

        try:
            # Check if we have a mapping for this symbol
            kraken_pair = self._symbol_mapping.get(symbol)
            if not kraken_pair:
                # Try direct mapping
                kraken_pair = symbol.replace("/", "")

            # Query Kraken for asset pairs
            response = await self._make_request("/0/public/AssetPairs")

            # Check if our pair exists
            valid = any(kraken_pair.upper() in key.upper() for key in response)

            # Cache result
            self._valid_symbols_cache[symbol] = {
                "valid": valid,
                "timestamp": datetime.now(UTC),
                "kraken_pair": kraken_pair if valid else None,
            }

        except Exception as e:
            self.logger.warning(
                f"Failed to validate symbol {symbol}: {e}",
                extra={"source_module": self._source_module},
            )
            # Default to true on error to avoid blocking
            return True
        else:
            return valid

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get detailed information about a trading pair from Kraken."""
        try:
            kraken_pair = self._map_to_kraken_pair(symbol)
            response = await self._make_request(
                "/0/public/AssetPairs",
                params={"pair": kraken_pair},
            )

            for key, info in response.items():
                if kraken_pair.upper() in key.upper():
                    return {
                        "symbol": symbol,
                        "kraken_pair": key,
                        "base": info.get("base"),
                        "quote": info.get("quote"),
                        "lot_decimals": info.get("lot_decimals"),
                        "pair_decimals": info.get("pair_decimals"),
                        "lot_multiplier": info.get("lot_multiplier"),
                        "leverage_buy": info.get("leverage_buy", []),
                        "leverage_sell": info.get("leverage_sell", []),
                        "fees": info.get("fees", []),
                        "fees_maker": info.get("fees_maker", []),
                        "fee_volume_currency": info.get("fee_volume_currency"),
                        "margin_call": info.get("margin_call"),
                        "margin_stop": info.get("margin_stop"),
                    }


        except Exception as e:
            self.logger.exception(
                f"Failed to get symbol info for {symbol}: ",
                extra={"source_module": self._source_module},
            )
            return {"symbol": symbol, "error": str(e)}
        else:
            return {"symbol": symbol, "error": "Pair not found"}

    def _map_to_kraken_pair(self, symbol: str) -> str:
        """Map internal symbol format to Kraken pair format."""
        # Check if we have a specific mapping
        if symbol in self._symbol_mapping:
            return self._symbol_mapping[symbol]

        # Try generic conversion (remove slash)
        return symbol.replace("/", "")

    def _convert_to_kraken_interval(self, frequency: str) -> int:
        """Convert frequency string to Kraken interval (in minutes)."""
        # Kraken intervals: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1w": 10080,
            "15d": 21600,
        }

        interval = mapping.get(frequency)
        if not interval:
            # Try to parse custom format
            if frequency.endswith("m"):
                try:
                    minutes = int(frequency[:-1])
                    # Round to nearest valid Kraken interval
                    valid_intervals = [1, 5, 15, 30, 60, 240, 1440, 10080, 21600]
                    interval = min(valid_intervals, key=lambda x: abs(x - minutes))
                except ValueError:
                    interval = 60  # Default to 1 hour
            else:
                interval = 60  # Default to 1 hour

        return interval

    def _validate_data_point(self, dp: HistoricalDataPoint) -> bool:
        """Validate OHLCV data integrity."""
        # Basic OHLCV validation
        if dp.high < dp.low:
            return False

        if dp.open <= 0 or dp.close <= 0:
            return False

        if dp.volume < 0:
            return False

        if dp.high < max(dp.open, dp.close):
            return False

        return not dp.low > min(dp.open, dp.close)

    def _generate_cache_key(self, request: DataRequest) -> str:
        """Generate cache key for request."""
        return f"{request.symbol}_{request.start_date.isoformat()}_{request.end_date.isoformat()}_{request.frequency}"

    def _is_cache_valid(self, cache_entry: dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if "timestamp" not in cache_entry:
            return False

        age = (datetime.now(UTC) - cache_entry["timestamp"]).total_seconds()
        return cast("bool", age < self._cache_ttl)

    def _is_cache_entry_valid(self, cache_entry: dict[str, Any]) -> bool:
        """Check if symbol cache entry is still valid."""
        if "timestamp" not in cache_entry:
            return False

        age = (datetime.now(UTC) - cache_entry["timestamp"]).total_seconds()
        return cast("bool", age < self._cache_ttl)

    def _cache_query_result(self, cache_key: str, data_points: list[HistoricalDataPoint]) -> None:
        """Cache query results with LRU eviction."""
        # Implement simple LRU by removing oldest entries
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = min(self._query_cache.keys(),
                           key=lambda k: self._query_cache[k].get("timestamp", datetime.min))
            del self._query_cache[oldest_key]

        self._query_cache[cache_key] = {
            "data": data_points,
            "timestamp": datetime.now(UTC),
        }

    def _update_api_metrics(self, success: bool, response_time: float) -> None:
        """Update API performance metrics."""
        self._api_metrics["total_requests"] += 1

        if success:
            self._api_metrics["successful_requests"] += 1
        else:
            self._api_metrics["failed_requests"] += 1

        # Update rolling average response time
        current_avg = self._api_metrics["average_response_time"]
        total_requests = self._api_metrics["total_requests"]

        self._api_metrics["average_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )

    async def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information about API provider health."""
        diagnostics = {
            "provider": "APIDataProvider",
            "status": "healthy",
            "api_url": self.base_url,
            "metrics": self._api_metrics.copy(),
            "cache_stats": {
                "query_cache_size": len(self._query_cache),
                "symbol_cache_size": len(self._valid_symbols_cache),
                "cache_hit_rate": (
                    self._api_metrics["cache_hits"] / max(1, self._api_metrics["total_requests"])
                ) if self._api_metrics["total_requests"] > 0 else 0,
            },
            "circuit_breaker": {
                "state": self._circuit_breaker.state,
                "failure_count": self._circuit_breaker.failure_count,
            },
            "rate_limiter": {
                "tokens_available": self._rate_limiter.tokens,
                "burst_limit": self._rate_limiter.burst,
            },
        }

        # Check if circuit breaker is open
        if self._circuit_breaker.is_open():
            diagnostics["status"] = "degraded"
            diagnostics["issues"] = ["Circuit breaker is open due to API failures"]

        return diagnostics
