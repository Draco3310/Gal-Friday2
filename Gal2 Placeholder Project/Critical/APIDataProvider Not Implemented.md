# APIDataProvider Implementation Design

**File**: `/gal_friday/providers/api_provider.py`
- **Line 20**: `self.logger.info("APIDataProvider not implemented")`
- **Issue**: Returns empty list instead of fetching from external APIs
- **Impact**: No external API data integration

## Overview
The APIDataProvider is a critical component that fetches historical market data from external APIs, specifically the Kraken exchange API. This implementation will be production-ready with proper error handling, rate limiting, data validation, and caching.

## Architecture Design

### 1. Core Components

```
APIDataProvider
├── Authentication & Configuration
│   ├── API credentials management
│   ├── Environment-specific configs
│   └── Security measures
├── HTTP Client Management
│   ├── Connection pooling
│   ├── Request/response handling
│   └── Retry logic
├── Rate Limiting
│   ├── Token bucket implementation
│   ├── Request queuing
│   └── Backpressure handling
├── Data Fetching
│   ├── OHLCV data retrieval
│   ├── Pagination handling
│   └── Time range management
├── Data Transformation
│   ├── Response parsing
│   ├── Data validation
│   └── Format conversion
└── Error Handling
    ├── Network errors
    ├── API errors
    └── Data quality issues
```

### 2. Key Features

1. **Multi-Source Support**: Initial Kraken implementation with extensibility for other exchanges
2. **Robust Error Handling**: Comprehensive error recovery with circuit breaker pattern
3. **Rate Limiting**: Respect API limits with intelligent request scheduling
4. **Data Validation**: Ensure data integrity and completeness
5. **Caching**: Reduce API calls for frequently requested data
6. **Monitoring**: Track API health, request metrics, and data quality

### 3. Integration Points

- Inherits from `HistoricalDataProvider` abstract base class
- Uses `DataRequest` and `HistoricalDataPoint` data structures
- Integrates with ConfigManager for API credentials
- Uses LoggerService for structured logging
- Publishes metrics to MonitoringService

## Implementation Plan

### Phase 1: Core Structure
```python
class APIDataProvider(HistoricalDataProvider):
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # API configuration
        self.base_url = config.get("api.base_url", "https://api.kraken.com")
        self.api_key = config.get("api.key")
        self.api_secret = config.get("api.secret")
        
        # HTTP client
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self._rate_limiter = TokenBucketRateLimiter(
            rate=config.get("api.rate_limit", 1),
            burst=config.get("api.burst_limit", 5)
        )
        
        # Circuit breaker for fault tolerance
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=APIError
        )
```

### Phase 2: Kraken-Specific Implementation

1. **OHLCV Data Endpoint**
   - Endpoint: `/0/public/OHLC`
   - Parameters: pair, interval, since
   - Response parsing and validation

2. **Authentication for Private Endpoints**
   - Use existing `generate_kraken_signature` utility
   - Secure credential management

3. **Data Transformation**
   - Convert Kraken timestamp format to datetime
   - Normalize price/volume data to Decimal
   - Map Kraken symbols to internal format

### Phase 3: Advanced Features

1. **Request Batching**
   - Combine multiple symbol requests
   - Optimize for API limits

2. **Incremental Updates**
   - Track last fetched timestamp
   - Fetch only new data

3. **Data Quality Checks**
   - Validate OHLCV relationships (High >= Low, etc.)
   - Check for gaps in timestamps
   - Detect anomalies in price/volume

## Pseudocode Implementation

```python
async def fetch_data(self, request: DataRequest) -> List[HistoricalDataPoint]:
    """Fetch historical data from Kraken API."""
    
    # Validate request
    if not await self.validate_symbol(request.symbol):
        raise ValueError(f"Unsupported symbol: {request.symbol}")
    
    # Check cache first
    cache_key = self._generate_cache_key(request)
    cached_data = await self._cache.get(cache_key)
    if cached_data and not self._is_stale(cached_data, request):
        return cached_data
    
    # Rate limit check
    await self._rate_limiter.acquire()
    
    # Circuit breaker check
    if self._circuit_breaker.is_open():
        raise APIError("Circuit breaker is open")
    
    try:
        # Prepare API request
        params = self._prepare_kraken_params(request)
        
        # Make API call with retry logic
        response = await self._make_request(
            endpoint="/0/public/OHLC",
            params=params,
            retries=3
        )
        
        # Parse and validate response
        data_points = self._parse_kraken_response(response, request)
        
        # Cache the results
        await self._cache.set(cache_key, data_points)
        
        # Record success metrics
        self._record_metrics(success=True, latency=response.elapsed)
        
        return data_points
        
    except Exception as e:
        self._circuit_breaker.record_failure()
        self._record_metrics(success=False, error=str(e))
        raise

async def _make_request(
    self, 
    endpoint: str, 
    params: Dict[str, Any], 
    retries: int = 3
) -> Dict[str, Any]:
    """Make HTTP request with retry logic."""
    
    last_error = None
    
    for attempt in range(retries):
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            url = f"{self.base_url}{endpoint}"
            
            async with self._session.get(
                url, 
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                # Check response status
                response.raise_for_status()
                
                # Parse JSON response
                data = await response.json()
                
                # Check for API errors
                if data.get("error"):
                    raise APIError(f"Kraken API error: {data['error']}")
                
                return data["result"]
                
        except aiohttp.ClientError as e:
            last_error = e
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                await asyncio.sleep(wait_time)
            continue
    
    raise last_error

def _parse_kraken_response(
    self, 
    response: Dict[str, Any], 
    request: DataRequest
) -> List[HistoricalDataPoint]:
    """Parse Kraken OHLC response into HistoricalDataPoint objects."""
    
    data_points = []
    
    # Extract OHLC data from response
    pair_data = response.get(self._map_to_kraken_pair(request.symbol), [])
    
    for candle in pair_data:
        # Kraken format: [time, open, high, low, close, vwap, volume, count]
        timestamp = datetime.fromtimestamp(candle[0], tz=UTC)
        
        # Skip if outside requested range
        if timestamp < request.start_date or timestamp > request.end_date:
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
                "source": "kraken"
            }
        )
        
        # Validate data point
        if self._validate_data_point(data_point):
            data_points.append(data_point)
        else:
            self.logger.warning(
                f"Invalid data point skipped: {data_point}",
                source_module=self._source_module
            )
    
    return data_points

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
    
    if dp.low > min(dp.open, dp.close):
        return False
    
    return True
```

## Testing Strategy

1. **Unit Tests**
   - Mock API responses
   - Test data transformation
   - Validate error handling

2. **Integration Tests**
   - Test with Kraken sandbox API
   - Verify rate limiting
   - Test circuit breaker

3. **Performance Tests**
   - Benchmark API response times
   - Test concurrent requests
   - Validate caching efficiency

## Monitoring & Observability

1. **Metrics to Track**
   - API request count/latency
   - Error rates by type
   - Cache hit/miss ratio
   - Data quality scores

2. **Alerts**
   - High error rates
   - Circuit breaker trips
   - Rate limit violations
   - Data quality degradation

## Security Considerations

1. **Credential Management**
   - Never log API secrets
   - Use environment variables
   - Implement key rotation

2. **Network Security**
   - Always use HTTPS
   - Validate SSL certificates
   - Implement request signing

## Future Enhancements

1. **Multi-Exchange Support**
   - Abstract exchange-specific logic
   - Implement factory pattern
   - Support Binance, Coinbase, etc.

2. **Advanced Features**
   - WebSocket support for real-time data
   - Order book snapshots
   - Trade history retrieval

3. **Data Enrichment**
   - Calculate additional indicators
   - Aggregate across timeframes
   - Cross-exchange arbitrage data