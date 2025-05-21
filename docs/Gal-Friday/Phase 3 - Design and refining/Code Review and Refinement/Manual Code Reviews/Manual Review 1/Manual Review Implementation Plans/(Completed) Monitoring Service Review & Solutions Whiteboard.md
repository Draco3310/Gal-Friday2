# **Monitoring Service (monitoring\_service.py): Review Summary & Solution Whiteboard**

## **1\. Summary of Manual Review Findings (monitoring\_services.md)**

* **Strengths:** Robust HALT state management (is\_halted), well-structured event publishing (SystemStateEvent), graceful shutdown (task cancellation, unsubscription), good basic error handling, clean asyncio usage for periodic checks.
* **Functional Requirements Gaps (High Priority):**
  * Missing monitoring for Kraken API connectivity (FR-901).
  * Missing monitoring for market data freshness (FR-902).
  * Missing monitoring for system resources (CPU/RAM) (FR-903).
  * Missing HALT triggers: Consecutive losses, critical API errors (partially covered by potential halt event?), market data staleness, excessive volatility (FR-905).
  * Missing configurable behavior for handling existing positions during HALT (FR-906).
* **Design & Implementation Issues:** Placeholder class usage, potential circular imports (handled via placeholders), inconsistent error handling approaches.
* **Configurability Issues:** Limited parameters are configurable; others (like specific HALT thresholds beyond drawdown) are missing. No clear config schema documented.
* **Documentation Gaps:** Minimal docstrings, lack of implementation guidance.

## **2\. Whiteboard: Proposed Solutions**

Focusing on implementing the missing monitoring checks and HALT triggers:

### **A. Implement Missing Monitoring Checks (High Priority \- FR-901, FR-902, FR-903)**

* **Problem:** The service only checks drawdown; other critical health aspects are ignored.
* **Solution:** Add separate async methods for each check, called within the main \_run\_periodic\_checks loop.
  1. **API Connectivity (\_check\_api\_connectivity):**
     * Periodically make a lightweight, authenticated API call to Kraken (e.g., /0/private/Balance).
     * Check for successful response (HTTP 200 and no API-level errors).
     * If fails consecutively (configurable threshold), log error and potentially trigger PotentialHaltTriggerEvent or directly call trigger\_halt. Requires access to API credentials or a way to interact with ExecutionHandler safely. *Alternative:* Rely on ExecutionHandler or DataIngestor to publish specific connectivity error events that MonitoringService subscribes to.
  2. **Market Data Freshness (\_check\_market\_data\_freshness):**
     * Requires access to the timestamp of the *last received* valid market data event (L2 or OHLCV) for relevant pairs. This state might need to be maintained within MonitoringService by subscribing to MARKET\_DATA\_L2/MARKET\_DATA\_OHLCV events or queried from DataIngestor/FeatureEngine if they expose such status.
     * Compare the last timestamp against datetime.utcnow().
     * If the age exceeds a configurable threshold (e.g., monitoring.data\_staleness\_threshold\_s), log error and trigger PotentialHaltTriggerEvent or trigger\_halt (FR-905).
  3. **System Resources (\_check\_system\_resources):**
     * Use a library like psutil (pip install psutil).
     * Periodically get psutil.cpu\_percent() and psutil.virtual\_memory().percent.
     * If usage exceeds configurable thresholds (e.g., monitoring.cpu\_threshold\_pct, monitoring.memory\_threshold\_pct) for a sustained period, log warnings. Critical thresholds could trigger a HALT.

\# In MonitoringService class

\# Add imports:
import psutil \# Requires installation
from datetime import datetime, timezone, timedelta
\# Need access to ExecutionHandler or a way to make test API calls
\# Need access to last data timestamps (e.g., via PubSub or direct query)

\# Add state in \_\_init\_\_ if needed:
\# self.\_last\_market\_data\_times: Dict\[str, datetime\] \= {} \# pair \-\> timestamp
\# self.\_consecutive\_api\_failures \= 0

async def \_run\_periodic\_checks(self) \-\> None:
    self.logger.info("MonitoringService periodic check task started.", source\_module=self.\_source)
    while True:
        try:
            await asyncio.sleep(self.\_check\_interval)
            if not self.\_is\_halted:
                self.logger.debug("Running periodic checks...", source\_module=self.\_source)
                \# \--- Call new checks \---
                await self.\_check\_drawdown()
                await self.\_check\_api\_connectivity() \# Add this
                await self.\_check\_market\_data\_freshness() \# Add this
                await self.\_check\_system\_resources() \# Add this
                \# \--- Add other checks \---
        \# ... (existing exception handling) ...

async def \_check\_api\_connectivity(self) \-\> None:
    """Placeholder: Checks connectivity to Kraken API."""
    \# This ideally needs access to ExecutionHandler's request method or similar
    \# For now, just a placeholder structure
    try:
        \# Example: Simulate making a lightweight call via ExecutionHandler
        \# success \= await self.\_execution\_handler.check\_api\_status() \# Needs method on ExecHandler
        success \= True \# Placeholder
        if success:
             \# self.\_consecutive\_api\_failures \= 0 \# Reset on success
             self.logger.debug("API connectivity check passed.", source\_module=self.\_source)
        else:
             \# self.\_consecutive\_api\_failures \+= 1
             \# failure\_threshold \= self.\_config.get\_int("monitoring.api\_failure\_threshold", 3\)
             \# if self.\_consecutive\_api\_failures \>= failure\_threshold:
             \#     reason \= f"API connectivity failed {self.\_consecutive\_api\_failures} consecutive times."
             \#     self.logger.error(reason, source\_module=self.\_source)
             \#     await self.trigger\_halt(reason=reason, source="AUTO: API Connectivity")
             pass \# Placeholder for failure logic
    except Exception as e:
        self.logger.error(f"Error during API connectivity check: {e}", source\_module=self.\_source, exc\_info=True)
        \# Increment failure count on exception too?
        \# self.\_consecutive\_api\_failures \+= 1 ... (handle threshold)

async def \_check\_market\_data\_freshness(self) \-\> None:
    """Checks if market data is recent enough."""
    \# Requires mechanism to know the last data timestamp per pair
    \# Example: Assuming self.\_last\_market\_data\_times is updated by subscribing to events
    staleness\_threshold\_s \= self.\_config.get\_float("monitoring.data\_staleness\_threshold\_s", 120.0)
    now \= datetime.now(timezone.utc)
    stale\_pairs \= \[\]

    \# Need to know which pairs \*should\* be active
    active\_pairs \= self.\_config.get\_list("trading.pairs", \[\]) \# Example: get from config

    for pair in active\_pairs:
         last\_ts \= self.\_last\_market\_data\_times.get(pair) \# Needs implementation
         if last\_ts is None:
              \# Handle case where data has never been received
              \# Maybe check how long the system has been running?
              self.logger.warning(f"No market data timestamp found for active pair {pair}.", source\_module=self.\_source)
              \# Consider triggering halt if no data after startup timeout
         elif (now \- last\_ts) \> timedelta(seconds=staleness\_threshold\_s):
              stale\_pairs.append(pair)
              self.logger.warning(f"Market data for {pair} is stale (last update: {last\_ts}, threshold: {staleness\_threshold\_s}s)", source\_module=self.\_source)

    if stale\_pairs:
         reason \= f"Market data stale for pairs: {', '.join(stale\_pairs)}"
         await self.trigger\_halt(reason=reason, source="AUTO: Market Data Staleness")

async def \_check\_system\_resources(self) \-\> None:
    """Monitors CPU and Memory usage."""
    try:
        cpu\_threshold \= self.\_config.get\_float("monitoring.cpu\_threshold\_pct", 90.0)
        mem\_threshold \= self.\_config.get\_float("monitoring.memory\_threshold\_pct", 90.0)

        cpu\_usage \= psutil.cpu\_percent(interval=None) \# Non-blocking
        mem\_usage \= psutil.virtual\_memory().percent

        self.logger.debug(f"System Resources: CPU={cpu\_usage:.1f}%, Memory={mem\_usage:.1f}%", source\_module=self.\_source)

        if cpu\_usage \> cpu\_threshold:
            self.logger.warning(f"High CPU usage detected: {cpu\_usage:.1f}% (Threshold: {cpu\_threshold}%)", source\_module=self.\_source)
            \# Consider triggering halt only if sustained high usage
        if mem\_usage \> mem\_threshold:
            self.logger.warning(f"High Memory usage detected: {mem\_usage:.1f}% (Threshold: {mem\_threshold}%)", source\_module=self.\_source)
            \# Consider triggering halt
            \# await self.trigger\_halt(reason=f"High Memory Usage: {mem\_usage:.1f}%", source="AUTO: System Resources")

    except Exception as e:
        self.logger.error(f"Error checking system resources: {e}", source\_module=self.\_source, exc\_info=True)

\# Need to add subscription to update self.\_last\_market\_data\_times in start()
\# async def \_update\_last\_data\_time(self, event: Union\[MarketDataL2Event, MarketDataOHLCVEvent\]):
\#    pair \= event.trading\_pair
\#    ts \= event.timestamp\_exchange or event.timestamp \# Prefer exchange time
\#    \# Ensure ts is timezone-aware UTC
\#    if ts.tzinfo is None: ts \= ts.replace(tzinfo=timezone.utc)
\#    self.\_last\_market\_data\_times\[pair\] \= ts

\# In start():
\# await self.pubsub.subscribe(EventType.MARKET\_DATA\_L2, self.\_update\_last\_data\_time)
\# await self.pubsub.subscribe(EventType.MARKET\_DATA\_OHLCV, self.\_update\_last\_data\_time)
\# In stop():
\# await self.pubsub.unsubscribe(EventType.MARKET\_DATA\_L2, self.\_update\_last\_data\_time)
\# await self.pubsub.unsubscribe(EventType.MARKET\_DATA\_OHLCV, self.\_update\_last\_data\_time)

### **B. Add Missing HALT Triggers (High Priority \- FR-905)**

* **Problem:** Only max drawdown triggers HALT. Consecutive losses, API errors, volatility, etc., are ignored.
* **Solution:**
  1. **Subscribe to Relevant Events:** Subscribe to ExecutionReportEvent to track losses. Subscribe to error events if other modules publish them (e.g., APIErrorEvent from ExecutionHandler).
  2. **Maintain State:** Keep track of consecutive losses per pair or globally. Track recent API error counts.
  3. **Implement Checks:** Add methods (\_check\_consecutive\_losses, \_check\_api\_errors, \_check\_market\_volatility) called in the periodic loop or triggered by relevant events.
  4. **Volatility Check:** Requires calculating price change percentage over a short interval (using data from MarketPriceService or recent OHLCV) and comparing against a configurable threshold.
  5. **Trigger Halt:** Call trigger\_halt with appropriate reason and source when thresholds are breached.

\# In MonitoringService class

\# Add state in \_\_init\_\_:
\# self.\_consecutive\_losses \= 0
\# self.\_recent\_api\_errors \= deque(maxlen=10) \# Store timestamps of recent errors

\# Add subscriptions in start():
\# await self.pubsub.subscribe(EventType.EXECUTION\_REPORT, self.\_handle\_execution\_report\_for\_losses)
\# await self.pubsub.subscribe(EventType.API\_ERROR, self.\_handle\_api\_error) \# Assuming custom APIErrorEvent

\# Add unsubscriptions in stop()

\# async def \_handle\_execution\_report\_for\_losses(self, event: ExecutionReportEvent):
\#     if event.order\_status \== "FILLED" and event.realized\_pnl \< 0: \# Needs PnL on report
\#          self.\_consecutive\_losses \+= 1
\#     elif event.order\_status \== "FILLED" and event.realized\_pnl \>= 0:
\#          self.\_consecutive\_losses \= 0 \# Reset on profit
\#     \# Check threshold
\#     loss\_limit \= self.\_config.get\_int("monitoring.consecutive\_loss\_limit", 5\)
\#     if self.\_consecutive\_losses \>= loss\_limit:
\#          reason \= f"Consecutive loss limit reached: {self.\_consecutive\_losses}"
\#          await self.trigger\_halt(reason=reason, source="AUTO: Consecutive Losses")

\# async def \_handle\_api\_error(self, event: APIErrorEvent): \# Needs custom event
\#      now \= time.time()
\#      self.\_recent\_api\_errors.append(now)
\#      \# Check frequency
\#      error\_threshold\_count \= self.\_config.get\_int("monitoring.api\_error\_threshold\_count", 5\)
\#      error\_threshold\_period\_s \= self.\_config.get\_int("monitoring.api\_error\_threshold\_period\_s", 60\)
\#      errors\_in\_period \= \[t for t in self.\_recent\_api\_errors if now \- t \< error\_threshold\_period\_s\]
\#      if len(errors\_in\_period) \>= error\_threshold\_count:
\#           reason \= f"High frequency of API errors: {len(errors\_in\_period)} in {error\_threshold\_period\_s}s"
\#           await self.trigger\_halt(reason=reason, source="AUTO: API Errors")

async def \_check\_market\_volatility(self) \-\> None:
    """Placeholder: Checks for excessive market volatility."""
    \# Requires price data access (e.g., via MarketPriceService or OHLCV history)
    \# 1\. Get recent price change (e.g., % change over last 5 mins)
    \# 2\. Compare change against config threshold (e.g., monitoring.volatility\_threshold\_pct)
    \# 3\. If exceeded, trigger halt
    pass \# Implementation depends on available price data mechanism

### **C. Implement Configurable HALT Position Behavior (High Priority \- FR-906)**

* **Problem:** The action taken on existing positions during a HALT isn't implemented or configurable.
* **Solution:**
  1. Modify trigger\_halt: After setting \_is\_halted \= True and publishing the state change event, check a configuration value (e.g., monitoring.halt.position\_behavior).
  2. If config is "close" or "liquidate":
     * Get open positions from PortfolioManager.
     * For each position, create and publish a "close position" command/event (needs a defined event type, e.g., ClosePositionCommand) to be handled by ExecutionHandler (placing market orders).
  3. If config is "maintain", do nothing with existing positions.

\# In MonitoringService.trigger\_halt method

async def trigger\_halt(self, reason: str, source: str) \-\> None:
    \# ... (set self.\_is\_halted \= True, log critical message) ...
    await self.\_publish\_state\_change("HALTED", reason, source)

    \# \--- Handle Existing Positions \---
    halt\_behavior \= self.\_config.get("monitoring.halt.position\_behavior", "maintain").lower()
    self.logger.info(f"HALT triggered. Position behavior set to: {halt\_behavior}", source\_module=self.\_source)

    if halt\_behavior \== "close" or halt\_behavior \== "liquidate":
         self.logger.warning("Attempting to close all open positions due to HALT.", source\_module=self.\_source)
         try:
              \# Need a way to get positions and trigger closure
              \# Option 1: Call PortfolioManager method
              \# open\_positions \= self.\_portfolio\_manager.get\_open\_positions() \# Needs method
              \# Option 2: Query PortfolioManager state
              current\_state \= self.\_portfolio\_manager.get\_current\_state()
              open\_positions \= current\_state.get("positions", {})

              if not open\_positions:
                   self.logger.info("No open positions found to close during HALT.", source\_module=self.\_source)
                   return

              for pair, pos\_data in open\_positions.items():
                   qty\_str \= pos\_data.get("quantity")
                   if not qty\_str: continue
                   qty \= Decimal(qty\_str)
                   if abs(qty) \> Decimal("1e-12"): \# Check if position exists
                        close\_side \= "SELL" if qty \> 0 else "BUY"
                        self.logger.info(f"Requesting closure of {pair} position ({close\_side} {abs(qty)})", source\_module=self.\_source)
                        \# Need a specific event/mechanism to tell ExecutionHandler to close
                        \# Example: Publish a new command event type
                        \# close\_command \= ClosePositionCommand(..., pair=pair, quantity=abs(qty), side=close\_side)
                        \# await self.\_pubsub.publish(close\_command)
                        \# For now, just log the intent
                   else:
                        self.logger.debug(f"Skipping zero quantity position for {pair}", source\_module=self.\_source)

         except Exception as e:
              self.logger.error(f"Error during attempt to close positions on HALT: {e}", source\_module=self.\_source, exc\_info=True)
    elif halt\_behavior \== "maintain":
         self.logger.info("Maintaining existing positions during HALT as per configuration.", source\_module=self.\_source)
    else:
         self.logger.warning(f"Unknown halt position behavior configured: {halt\_behavior}. Maintaining positions.", source\_module=self.\_source)

### **D. Improve Configuration & Placeholders (Medium Priority)**

* **Problem:** Limited configurability, placeholder classes used at runtime if imports fail, hardcoded event types in placeholders.
* **Solution:**
  * **Configuration:** Add configuration keys for all thresholds (consecutive losses, API errors, volatility, data staleness, resource usage) and the HALT position behavior. Load these in \_\_init\_\_ with validation and clear logging if defaults are used.
  * **Placeholders:** Remove runtime placeholder classes. Rely on if TYPE\_CHECKING: for type hints only. Ensure main.py handles missing module imports gracefully during startup, preventing the service from running without its dependencies. The placeholder logic in monitoring\_service.py itself should be removed.
  * **Event Types:** Import EventType directly from core.events instead of using string literals or placeholder classes.

**Conclusion:** Implementing the missing monitoring checks (API, data, resources) and HALT triggers (losses, errors, volatility) is crucial for meeting the functional requirements (FR-9xx). Making HALT behavior configurable and improving configuration management will enhance robustness and usability. Fixing the placeholder/import issues is necessary for stable runtime behavior.
