# **Data Ingestor: Review Summary & Solution Whiteboard**

## **1\. Summary of Manual Review Findings (data\_ingestor.md)**

The review identified strengths and weaknesses in the data\_ingestor.py module:

* **Strengths:** Robust service lifecycle (start/stop), good basic error handling (especially in message processing), clean event publishing, effective use of asyncio, configurable parameters, and good type hinting.
* **Exchange API Integration:**
  * **Insufficient Rate Limit Handling:** Doesn't track or respect Kraken API limits.
  * **Limited WebSocket Error Handling:** Simple reconnection delay, no exponential backoff, doesn't handle specific WS error codes.
  * **Incomplete Authentication:** No support for authenticated feeds.
  * **No Heartbeat Monitoring:** Doesn't explicitly check for Kraken's heartbeat messages.
* **Data Processing & Validation:**
  * **Inadequate Response Validation:** Limited checks on the structure of API responses.
  * **Missing Data Sanity Checks:** No checks for anomalies (e.g., unreasonable prices/volumes).
  * **Incomplete Error Classification:** Doesn't distinguish between transient and permanent errors for better retry logic.
* **Architectural Concerns:**
  * **Hardcoded Kraken Logic:** Exchange-specific details are embedded directly.
  * **Tight Coupling:** Assumes specific event structures directly.
  * **No Configuration Validation:** Doesn't check for required config parameters at startup.
* **Performance & Scaling:**
  * **No Throttling Mechanism:** REST calls aren't throttled.
  * **Limited Connection Pooling:** aiohttp session could be better configured.
  * **Memory Management Concerns:** Potential memory usage from large order book snapshots.

## **2\. Whiteboard: Proposed Solutions**

Here's a breakdown of solutions addressing the high and medium priority recommendations:

### **A. Rate Limit Tracking (High Priority)**

* **Problem:** Risk of hitting API rate limits, leading to temporary bans. Affects REST calls primarily (if any were used \- the current code focuses on WebSocket, but this is good practice). The WebSocket itself might have connection or message limits.
* **Solution:** Implement a helper class or mechanism to track request timestamps and enforce delays if the rate limit is approached. The review suggests a RateLimitTracker class. For WebSockets, this is less about request *rate* and more about connection limits or message complexity limits, which are harder to track proactively. Focus on robust connection handling instead. *If REST calls were added*, the tracker would be essential.\*
  \# Example RateLimitTracker (if REST calls were used)
  import time
  import asyncio
  from collections import deque

  class RateLimitTracker:
      def \_\_init\_\_(self, max\_requests: int, per\_seconds: int):
          self.max\_requests \= max\_requests
          self.per\_seconds \= per\_seconds
          self.request\_times \= deque()
          self.\_lock \= asyncio.Lock()

      async def wait\_if\_needed(self):
          async with self.\_lock:
              now \= time.monotonic()
              \# Remove timestamps older than the window
              while self.request\_times and self.request\_times\[0\] \<= now \- self.per\_seconds:
                  self.request\_times.popleft()

              if len(self.request\_times) \>= self.max\_requests:
                  wait\_time \= self.per\_seconds \- (now \- self.request\_times\[0\])
                  if wait\_time \> 0:
                      self.\_logger.debug(f"Rate limit approached. Waiting for {wait\_time:.2f}s")
                      await asyncio.sleep(wait\_time)
              \# Record the current request time \*after\* waiting
              self.request\_times.append(time.monotonic())

  \# Usage (within a hypothetical REST request method):
  \# await self.rate\_limiter.wait\_if\_needed()
  \# response \= await self.session.get(...)

### **B. Enhanced WebSocket Error Handling (High Priority)**

* **Problem:** Simple fixed delay on reconnection doesn't handle persistent temporary issues well and can hammer the server. Doesn't classify errors.
* **Solution:** Implement exponential backoff with jitter for reconnection attempts. Modify the main connection loop (start or a dedicated \_reconnect method) to use this strategy. Classify errors (transient vs. permanent) if possible based on WebSocket close codes or specific exceptions.
  \# In DataIngestor class
  import random

  async def \_reconnect\_with\_backoff(self):
      """Attempts reconnection with exponential backoff and jitter."""
      retry\_count \= 0
      \# Get parameters from config, provide defaults
      max\_retries \= self.\_config.get\_int("websocket.max\_retries", 5\) \# Example config key
      base\_delay \= self.\_config.get\_float("websocket.base\_delay\_seconds", 2.0) \# Example config key
      max\_delay \= self.\_config.get\_float("websocket.max\_delay\_seconds", 60.0) \# Example config key

      while self.\_is\_running and retry\_count \< max\_retries:
          retry\_count \+= 1
          delay \= min(base\_delay \* (2 \*\* (retry\_count \- 1)), max\_delay)
          jitter \= random.uniform(0, delay \* 0.1) \# Add up to 10% jitter
          total\_delay \= delay \+ jitter

          self.logger.warning(
              f"WebSocket disconnected. Attempting reconnect {retry\_count}/{max\_retries} "
              f"in {total\_delay:.2f} seconds...",
              source\_module=self.\_source\_module
          )
          await asyncio.sleep(total\_delay)

          if not self.\_is\_running: break \# Check if stop was called during sleep

          \# Try to establish connection again
          if await self.\_establish\_connection():
               \# If successful, try to setup (subscribe, etc.)
               subscription\_msg \= self.\_build\_subscription\_message()
               if subscription\_msg and await self.\_setup\_connection(subscription\_msg):
                   self.logger.info("WebSocket reconnected and setup successfully.")
                   return True \# Reconnect successful

      self.logger.error(
          f"Failed to reconnect WebSocket after {max\_retries} attempts. Stopping.",
          source\_module=self.\_source\_module
      )
      self.\_is\_running \= False \# Stop the main loop
      return False

  \# Modify the main 'start' loop:
  async def start(self) \-\> None:
      \# ... (initial setup) ...
      while self.\_is\_running:
          connected\_and\_setup \= False
          try:
              if await self.\_establish\_connection():
                  subscription\_msg \= self.\_build\_subscription\_message()
                  if subscription\_msg and await self.\_setup\_connection(subscription\_msg):
                      connected\_and\_setup \= True
                      await self.\_message\_listen\_loop() \# Normal operation
                  else:
                       await self.\_cleanup\_connection() \# Setup failed
              \# If establish or setup failed, connected\_and\_setup remains False
          except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError) as e:
               self.logger.warning(f"WebSocket connection closed: {e.code} {e.reason}", source\_module=self.\_source\_module)
               \# Expected closure or error, proceed to reconnect logic
          except Exception as e:
              self.logger.error(f"Unexpected error in main loop: {e}", source\_module=self.\_source\_module, exc\_info=True)
              \# Unexpected error, proceed to reconnect logic
          finally:
               await self.\_cleanup\_connection() \# Ensure cleanup before potential reconnect

          \# Reconnect logic only if running and connection failed/closed
          if self.\_is\_running and not connected\_and\_setup:
               if not await self.\_reconnect\_with\_backoff():
                   break \# Stop if reconnect fails permanently

      self.logger.info("Data Ingestor stopped.", source\_module=self.\_\_class\_\_.\_\_name\_\_)

### **C. Data Validation (High Priority)**

* **Problem:** Malformed messages from the exchange could cause errors or propagate bad data.
* **Solution:** Add explicit validation checks within the message handling methods (\_handle\_book\_data, \_handle\_ohlc\_data, etc.) *before* processing the data. Check for expected keys, data types, and potentially value ranges (e.g., price \> 0). Libraries like Pydantic could be used for more complex schema validation.
  \# Example within \_handle\_book\_data
  async def \_handle\_book\_data(self, data: dict) \-\> bool:
      \# \--- Start Validation \---
      if not isinstance(data.get("data"), list):
           self.logger.warning(f"Invalid book message: 'data' is not a list. Msg: {str(data)\[:200\]}", source\_module=self.\_source\_module)
           return False \# Skip processing

      msg\_type \= data.get("type")
      if msg\_type not in \["snapshot", "update"\]:
           self.logger.warning(f"Invalid book message type: {msg\_type}. Msg: {str(data)\[:200\]}", source\_module=self.\_source\_module)
           return False \# Skip processing
      \# \--- End Validation \---

      is\_snapshot \= msg\_type \== "snapshot"
      processed\_ok \= True
      for book\_item in data\["data"\]:
           \# \--- Start Item Validation \---
           if not isinstance(book\_item, dict):
               self.logger.warning(f"Invalid book item: not a dict. Item: {str(book\_item)\[:200\]}", source\_module=self.\_source\_module)
               processed\_ok \= False
               continue \# Skip this item

           symbol \= book\_item.get("symbol")
           if not isinstance(symbol, str) or not symbol:
               self.logger.warning(f"Book item missing/invalid symbol. Item: {str(book\_item)\[:200\]}", source\_module=self.\_source\_module)
               processed\_ok \= False
               continue \# Skip this item

           \# Validate bids/asks structure (list of dicts with price/qty)
           for side\_key in \["asks", "bids"\]:
               side\_data \= book\_item.get(side\_key)
               if side\_data is not None: \# It's okay if a side is missing in an update
                   if not isinstance(side\_data, list):
                        self.logger.warning(f"Book item '{side\_key}' is not a list for {symbol}. Item: {str(book\_item)\[:200\]}", source\_module=self.\_source\_module)
                        processed\_ok \= False; break \# Stop validating this item
                   for level in side\_data:
                        if not isinstance(level, dict) or "price" not in level or "qty" not in level:
                            self.logger.warning(f"Invalid level in '{side\_key}' for {symbol}. Level: {str(level)\[:100\]}", source\_module=self.\_source\_module)
                            processed\_ok \= False; break \# Stop validating this item
               if not processed\_ok: break \# Stop validating item if inner loop failed
           if not processed\_ok: continue \# Skip processing this item
           \# \--- End Item Validation \---

           \# ... (rest of the processing logic: \_apply\_book\_snapshot/\_update, \_truncate, \_validate\_checksum, \_publish) ...
           \# Pass the symbol and book\_item to these functions

      return processed\_ok \# Return overall status for the message

### **D. Heartbeat Monitoring (Medium Priority)**

* **Problem:** The connection might appear open but unresponsive (silent failure). The current liveness check relies on *any* message, not specifically heartbeats.
* **Solution:** Modify the \_handle\_channel\_message to specifically recognize Kraken's heartbeat messages. Reset a dedicated \_last\_heartbeat\_received\_time timestamp. Modify \_monitor\_connection\_liveness\_loop to check *this specific timestamp* in addition to the general message timestamp, providing a more reliable check against silent connection hangs.
  \# In \_\_init\_\_
  self.\_last\_heartbeat\_received\_time: Optional\[datetime\] \= None

  \# In \_handle\_channel\_message
  elif channel \== "heartbeat":
      self.logger.debug("Received heartbeat.", source\_module=self.\_\_class\_\_.\_\_name\_\_)
      self.\_last\_heartbeat\_received\_time \= datetime.now(timezone.utc) \# Update specific timestamp
      \# No need to update self.\_last\_message\_received\_time here, as the main loop does it

  \# In \_monitor\_connection\_liveness\_loop
  async def \_monitor\_connection\_liveness\_loop(self) \-\> None:
      \# ... (setup) ...
      max\_heartbeat\_interval\_s \= self.\_config.get\_float("websocket.max\_heartbeat\_interval\_s", 60.0) \# Example config

      while self.\_is\_running and self.\_connection and not self.\_connection.closed:
          await asyncio.sleep(check\_interval)
          now \= datetime.now(timezone.utc)
          general\_timeout \= False
          heartbeat\_timeout \= False

          \# Check general message timeout
          if self.\_last\_message\_received\_time:
              time\_since\_last \= now \- self.\_last\_message\_received\_time
              if time\_since\_last \> timedelta(seconds=self.\_connection\_timeout):
                  self.logger.warning(
                      f"No messages received for {time\_since\_last.total\_seconds():.1f}s "
                      f"(\> {self.\_connection\_timeout}s timeout).",
                      source\_module=self.\_\_class\_\_.\_\_name\_\_
                  )
                  general\_timeout \= True
          else: \# No messages received \*at all\* yet after connecting
               if now \- self.\_connection\_established\_time \> timedelta(seconds=self.\_connection\_timeout): \# Need to store connection time
                    self.logger.warning(f"No messages received within {self.\_connection\_timeout}s of connecting.")
                    general\_timeout \= True

          \# Check specific heartbeat timeout
          if self.\_last\_heartbeat\_received\_time:
               time\_since\_last\_hb \= now \- self.\_last\_heartbeat\_received\_time
               if time\_since\_last\_hb \> timedelta(seconds=max\_heartbeat\_interval\_s):
                    self.logger.warning(
                        f"No heartbeat received for {time\_since\_last\_hb.total\_seconds():.1f}s "
                        f"(\> {max\_heartbeat\_interval\_s}s timeout).",
                        source\_module=self.\_\_class\_\_.\_\_name\_\_
                    )
                    heartbeat\_timeout \= True
          \# else: \# No heartbeats received yet, maybe okay early on? Depends on exchange.

          \# Trigger reconnect if either timeout occurs
          if general\_timeout or heartbeat\_timeout:
               self.logger.warning("Liveness check failed. Triggering reconnect.")
               if self.\_connection and not self.\_connection.closed:
                   \# Use create\_task to avoid blocking the monitor loop
                   asyncio.create\_task(self.\_cleanup\_connection())
               break \# Exit monitor loop, main loop will handle reconnect
      \# ... (cleanup) ...

### **E. Exchange Abstraction Layer (Medium Priority)**

* **Problem:** Code is tightly coupled to Kraken's specific API format and behavior. Adding another exchange would require significant refactoring.
* **Solution:** Define base classes or interfaces (ExchangeWebsocketInterface, ExchangeRestInterface) outlining common methods (connect, subscribe\_book, parse\_book\_message, get\_ohlcv, etc.). Implement Kraken-specific versions (KrakenWebsocketAdapter, KrakenRestAdapter). The DataIngestor would then use the appropriate adapter based on configuration. This is a larger refactoring effort.
  \# \--- interfaces.py \---
  from abc import ABC, abstractmethod
  from typing import List, Dict, Any, Callable, Coroutine

  class ExchangeWebsocketInterface(ABC):
      @abstractmethod
      async def connect(self, url: str, message\_handler: Callable\[\[Dict\], Coroutine\]):
          pass
      @abstractmethod
      async def subscribe(self, subscriptions: List\[Dict\]):
          pass
      @abstractmethod
      async def close(self):
          pass
      \# Potentially add parse methods if standardizing parsing logic

  \# \--- kraken\_adapter.py \---
  class KrakenWebsocketAdapter(ExchangeWebsocketInterface):
      \# Implementation using websockets library for Kraken specifics
      async def connect(self, url: str, message\_handler: Callable\[\[Dict\], Coroutine\]):
           \# ... connect logic ...
           \# Start listen loop calling message\_handler
           pass
      async def subscribe(self, subscriptions: List\[Dict\]):
           \# ... build Kraken subscribe message and send ...
           pass
      async def close(self):
           \# ... close logic ...
           pass

  \# \--- data\_ingestor.py (Refactored) \---
  class DataIngestor:
      def \_\_init\_\_(self, config: "ConfigManager", pubsub\_manager: "PubSubManager", logger\_service: LoggerService):
          \# ...
          self.\_exchange\_adapter \= self.\_create\_exchange\_adapter(config) \# Factory method
          \# ...

      def \_create\_exchange\_adapter(self, config) \-\> ExchangeWebsocketInterface:
           exchange\_name \= config.get("exchange.name", "kraken").lower()
           if exchange\_name \== "kraken":
               return KrakenWebsocketAdapter(config, self.logger) \# Pass config/logger
           \# elif exchange\_name \== "binance":
           \#     return BinanceWebsocketAdapter(config, self.logger)
           else:
               raise ValueError(f"Unsupported exchange: {exchange\_name}")

      async def start(self):
           \# ...
           url \= self.\_config.get("exchange.websocket\_url") \# Get URL from config
           await self.\_exchange\_adapter.connect(url, self.\_handle\_parsed\_message)
           subscriptions \= self.\_build\_subscriptions() \# Generic format?
           await self.\_exchange\_adapter.subscribe(subscriptions)
           \# ... main loop might just wait for adapter tasks ...

      async def \_handle\_parsed\_message(self, message: Dict):
           \# Handler passed to the adapter, receives already parsed messages
           \# Contains logic to route to \_handle\_book, \_handle\_ohlc etc.
           \# This method now contains the \*standardized\* data handling
           pass
      \# ... other methods adapt to use the adapter ...

### **F. Request Throttling (Medium Priority)**

* **Problem:** If REST API calls were used (e.g., for initial OHLCV fetch or other data), they could hit rate limits without throttling.
* **Solution:** Use a throttling mechanism similar to the RateLimitTracker (or potentially the same class if limits are similar) before making any REST API calls using aiohttp (if it were being used).
  \# Assume self.throttler \= RequestThrottler(...) exists
  \# Assume self.http\_session \= aiohttp.ClientSession(...) exists

  async def fetch\_ohlcv\_rest(self, pair: str, interval: int, since: Optional\[int\] \= None):
      \# Example hypothetical REST call
      await self.throttler.acquire() \# Wait if needed before making the call
      api\_url \= self.\_config.get("kraken.api\_url") \# Example config
      endpoint \= f"{api\_url}/public/OHLC"
      params \= {'pair': pair, 'interval': interval}
      if since:
          params\['since'\] \= since

      try:
           async with self.http\_session.get(endpoint, params=params) as response:
               response.raise\_for\_status() \# Check for HTTP errors
               data \= await response.json()
               \# ... process data ...
               return data
      except aiohttp.ClientError as e:
           self.logger.error(f"HTTP error fetching OHLCV for {pair}: {e}", source\_module=self.\_source\_module)
           return None
      except Exception as e:
           self.logger.error(f"Error processing OHLCV REST response for {pair}: {e}", source\_module=self.\_source\_module, exc\_info=True)
           return None

This whiteboard provides a plan for addressing the key findings from the code review. Implementing these changes, particularly the abstraction layer, would significantly improve the robustness and maintainability of the DataIngestor.
