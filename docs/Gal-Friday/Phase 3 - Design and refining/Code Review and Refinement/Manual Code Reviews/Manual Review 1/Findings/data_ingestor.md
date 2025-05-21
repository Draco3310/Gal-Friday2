# Manual Code Review Findings: `data_ingestor.py`

## Review Date: May 5, 2025
## Reviewer: AI Assistant
## File Reviewed: `src/gal_friday/data_ingestor.py`

## Summary

The `data_ingestor.py` module serves as the primary interface for market data acquisition, responsible for connecting to the Kraken exchange API, retrieving both Level 2 (order book) and OHLCV (candle) data, and publishing this data to the internal event system. The implementation shows solid handling of websocket connections, OHLCV data retrieval, and robust error recovery mechanisms.

However, there are several areas of concern including insufficient API rate limit handling, incomplete validation of exchange responses, and limited handling of reconnection scenarios. The module also lacks comprehensive documentation of its exchange-specific behaviors and assumptions.

## Strengths

1. **Robust Service Lifecycle Management**: Clear start/stop methods with proper resource cleanup and graceful shutdown handling.

2. **Comprehensive Error Handling**: Good use of try/except blocks with appropriate logging and recovery mechanisms, particularly in the websocket message processing.

3. **Clean Event Publishing**: Consistent transformation of exchange data into internal event formats with proper metadata.

4. **Effective Use of asyncio**: Good implementation of concurrent websocket and REST API calls with proper task management.

5. **Configurable Behavior**: Key parameters like polling intervals and reconnection delays are configurable via the configuration manager.

6. **Type Safety**: Good use of type hints throughout the codebase for improved code clarity and safety.

## Issues Identified

### A. Exchange API Integration

1. **Insufficient Rate Limit Handling**: The implementation doesn't track or respect Kraken's API rate limits, potentially leading to temporary IP bans during high-frequency polling or connection issues.

2. **Limited WebSocket Error Handling**: The websocket reconnection logic doesn't implement exponential backoff, and doesn't handle specific websocket error codes with appropriate responses.

3. **Incomplete Authentication**: The websocket connection doesn't implement authenticated subscriptions for private feeds that might be needed for future features.

4. **No Heartbeat Monitoring**: The implementation doesn't explicitly monitor heartbeat messages from Kraken to detect silent connection failures.

### B. Data Processing & Validation

1. **Inadequate Response Validation**: Limited validation of response structures from the Kraken API, potentially allowing malformed data to propagate through the system.

2. **Missing Data Sanity Checks**: No validation of price/volume ranges or other anomaly detection that could identify erroneous market data.

3. **Incomplete Error Classification**: The error handling doesn't differentiate between transient errors (which should trigger retries) and permanent errors (which should trigger alerts).

### C. Architectural Concerns

1. **Hardcoded Kraken-Specific Logic**: The module contains Kraken-specific message formats and API endpoints without a clear abstraction layer for potential multi-exchange support.

2. **Tight Coupling with Event System**: The module assumes specific event structures without a layer of abstraction that would allow for changes in the event system.

3. **No Configuration Validation**: The module doesn't validate that all required configuration parameters are present at startup.

### D. Performance & Scaling

1. **No Throttling Mechanism**: Missing implementation of request throttling for REST API calls to prevent overwhelming the exchange or hitting rate limits.

2. **Limited Connection Pooling**: The aiohttp session isn't optimally configured for connection pooling which could impact performance during heavy API usage.

3. **Memory Management Concerns**: The order book snapshots could potentially consume significant memory during high-volume trading periods without proper size limiting.

## Recommendations

### High Priority

1. **Implement Rate Limit Tracking**: Add a rate limit tracking mechanism that respects Kraken's published API limits:
   ```python
   class RateLimitTracker:
       def __init__(self, limit_per_second: int = 1):
           self.limit_per_second = limit_per_second
           self.requests = []

       async def wait_if_needed(self):
           """Wait if we're approaching the rate limit."""
           now = time.time()
           # Remove old requests from tracking
           self.requests = [r for r in self.requests if r > now - 1]

           if len(self.requests) >= self.limit_per_second:
               # Wait until we're under the limit
               sleep_time = 1 - (now - self.requests[0])
               if sleep_time > 0:
                   await asyncio.sleep(sleep_time)

           self.requests.append(time.time())
   ```

2. **Enhance WebSocket Error Handling**: Implement exponential backoff and proper error classification:
   ```python
   async def _reconnect_websocket(self):
       """Reconnect to the websocket with exponential backoff."""
       retry_count = 0
       max_retries = self.config.get_int("websocket.max_retries", 10)
       base_delay = self.config.get_float("websocket.base_delay_seconds", 1.0)

       while retry_count < max_retries:
           try:
               delay = base_delay * (2 ** retry_count)
               jitter = random.uniform(0, 0.1 * delay)
               total_delay = delay + jitter

               self.logger.info(
                   f"Reconnecting websocket in {total_delay:.2f} seconds "
                   f"(attempt {retry_count + 1}/{max_retries})",
                   source_module=self.__class__.__name__
               )

               await asyncio.sleep(total_delay)
               await self._connect_websocket()
               return  # Success
           except WebSocketError as e:
               if e.is_permanent():
                   self.logger.error(
                       f"Permanent websocket error: {str(e)}. Stopping reconnection attempts.",
                       source_module=self.__class__.__name__
                   )
                   break

               retry_count += 1
   ```

3. **Add Data Validation**: Implement comprehensive validation of exchange responses before processing:
   ```python
   def _validate_orderbook_message(self, message: dict) -> bool:
       """Validate an orderbook message from Kraken."""
       try:
           if not isinstance(message, list) or len(message) < 2:
               self.logger.warning(
                   f"Invalid orderbook message format: {message}",
                   source_module=self.__class__.__name__
               )
               return False

           channel_name = message[2]
           if channel_name != "book":
               self.logger.warning(
                   f"Unexpected channel name for orderbook: {channel_name}",
                   source_module=self.__class__.__name__
               )
               return False

           data = message[1]
           # Check for required fields in the data structure
           if not isinstance(data, dict):
               self.logger.warning(
                   f"Invalid orderbook data format: {data}",
                   source_module=self.__class__.__name__
               )
               return False

           # Validate specific fields based on Kraken's documentation
           # [Additional validation logic here]

           return True
       except Exception as e:
           self.logger.error(
               f"Error validating orderbook message: {str(e)}",
               source_module=self.__class__.__name__,
               exc_info=True
           )
           return False
   ```

### Medium Priority

1. **Implement Heartbeat Monitoring**: Add explicit monitoring of Kraken's heartbeat messages:
   ```python
   async def _monitor_heartbeats(self):
       """Monitor heartbeats from Kraken WebSocket."""
       last_heartbeat = time.time()
       max_heartbeat_interval = self.config.get_int(
           "websocket.max_heartbeat_interval_seconds",
           30
       )

       while self._ws and not self._ws.closed:
           now = time.time()
           if now - last_heartbeat > max_heartbeat_interval:
               self.logger.warning(
                   f"No heartbeat received in {max_heartbeat_interval} seconds. "
                   "Reconnecting websocket.",
                   source_module=self.__class__.__name__
               )
               await self._reconnect_websocket()
               last_heartbeat = time.time()

           await asyncio.sleep(5)  # Check every 5 seconds
   ```

2. **Create Exchange Abstraction Layer**: Refactor Kraken-specific logic into adapter classes:
   ```python
   class ExchangeAdapter:
       """Base class for exchange-specific adapters."""

       async def connect_websocket(self) -> None:
           """Connect to the exchange websocket."""
           raise NotImplementedError

       async def subscribe_to_orderbook(self, trading_pair: str) -> None:
           """Subscribe to orderbook updates for a trading pair."""
           raise NotImplementedError

       # Additional exchange-specific methods

   class KrakenExchangeAdapter(ExchangeAdapter):
       """Kraken-specific implementation of the exchange adapter."""

       async def connect_websocket(self) -> None:
           # Kraken-specific websocket connection logic
           pass

       async def subscribe_to_orderbook(self, trading_pair: str) -> None:
           # Kraken-specific subscription logic
           pass
   ```

3. **Implement Request Throttling**: Add throttling for REST API calls:
   ```python
   class RequestThrottler:
       """Throttles API requests to stay within rate limits."""

       def __init__(self, max_requests_per_second: int):
           self.max_requests_per_second = max_requests_per_second
           self.request_times = []
           self._lock = asyncio.Lock()

       async def acquire(self):
           """Acquire permission to make a request, waiting if necessary."""
           async with self._lock:
               now = time.time()
               # Remove old requests
               self.request_times = [t for t in self.request_times if t > now - 1]

               if len(self.request_times) >= self.max_requests_per_second:
                   # Calculate wait time
                   wait_time = 1 - (now - self.request_times[0])
                   if wait_time > 0:
                       await asyncio.sleep(wait_time)

               # Add current request time
               self.request_times.append(time.time())
   ```

### Low Priority

1. **Add Configuration Validation**: Implement validation of required configuration parameters:
   ```python
   def _validate_configuration(self) -> bool:
       """Validate that all required configuration parameters are present."""
       required_params = [
           "kraken.api_url",
           "kraken.websocket_url",
           "trading.pairs",
           "polling.ohlcv_interval_seconds"
       ]

       missing_params = []
       for param in required_params:
           if self.config.get(param) is None:
               missing_params.append(param)

       if missing_params:
           self.logger.error(
               f"Missing required configuration parameters: {', '.join(missing_params)}",
               source_module=self.__class__.__name__
           )
           return False

       return True
   ```

2. **Optimize Connection Pooling**: Improve aiohttp session configuration:
   ```python
   def _create_session(self) -> aiohttp.ClientSession:
       """Create an optimized aiohttp session for API requests."""
       timeout = aiohttp.ClientTimeout(
           total=self.config.get_int("http.timeout_seconds", 30),
           connect=self.config.get_int("http.connect_timeout_seconds", 5)
       )

       connector = aiohttp.TCPConnector(
           limit=self.config.get_int("http.max_connections", 100),
           ttl_dns_cache=self.config.get_int("http.dns_cache_ttl_seconds", 300),
           ssl=False  # Kraken uses SSL by default in their URL
       )

       return aiohttp.ClientSession(
           timeout=timeout,
           connector=connector,
           headers={"User-Agent": "Gal-Friday/0.1"}
       )
   ```

3. **Enhance Documentation**: Add comprehensive documentation of exchange-specific behaviors and assumptions:
   ```python
   class DataIngestor:
       """
       Handles market data acquisition from Kraken exchange.

       Exchange-Specific Behavior:
       - Kraken WebSocket provides L2 depth data with up to 10 levels by default
       - Order book snapshots are provided on initial subscription, then updates
       - OHLCV data is retrieved via REST API, with candles returned in reverse chronological order
       - Kraken rate limits: 15 requests per minute for REST API without API key
       - WebSocket connections may be terminated by the server after prolonged inactivity

       Assumptions:
       - Trading pairs are specified in format XBT/USD in config but converted to Kraken format (XBTUSD)
       - System will operate with the limited depth provided by the public feed
       - Candle data poll interval should be slightly longer than the candle interval
         (e.g., 70 seconds for 1-minute candles)
       """
   ```

## Compliance Assessment

The `data_ingestor.py` module partially complies with the requirements specified in the interface definitions document:

1. **Fully Compliant**: The module correctly publishes market data events as specified in sections 3.1 and 3.2 of the [inter_module_comm](../../../../../Phase%201%20-%20Requirements%20Analysis%20%26%20Planning/inter_module_comm_gal_friday_v0.1.md) document.

2. **Partially Compliant**: The module implements WebSocket connection handling but lacks robust error recovery and heartbeat monitoring as implied by NFR-301 and NFR-302.

3. **Non-Compliant**: The module doesn't fully implement rate limit handling as required by NFR-304, potentially leading to API access issues.

4. **Fully Compliant**: The module correctly handles the service lifecycle (start/stop) as required by the interface definition.

5. **Partially Compliant**: The module supports the required trading pairs (XRP/USD, DOGE/USD) per NFR-901 but doesn't validate their existence in configuration.

The most critical gaps are in the rate limit handling and WebSocket error recovery mechanisms, which could impact system stability during connection issues or high API usage.

## Follow-up Actions

- [ ] Implement rate limit tracking and adherence mechanism
- [ ] Enhance WebSocket reconnection with exponential backoff
- [ ] Add comprehensive validation of exchange responses
- [ ] Implement heartbeat monitoring for WebSocket connections
- [ ] Create abstraction layer for exchange-specific logic
- [ ] Add configuration validation at startup
- [ ] Optimize HTTP connection handling for better performance
- [ ] Enhance documentation of exchange-specific behaviors
