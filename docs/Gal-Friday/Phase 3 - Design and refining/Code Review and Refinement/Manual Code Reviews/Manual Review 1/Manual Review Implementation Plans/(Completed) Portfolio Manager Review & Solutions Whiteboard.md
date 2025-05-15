# **Portfolio Manager (portfolio\_manager.py): Review Summary & Solution Whiteboard**

## **1\. Summary of Manual Review Findings (portfolio\_manager.md)**

* **Strengths:** Robust financial calculations (Decimal), comprehensive state management (positions, cash, equity), good event consumption (ExecutionReportEvent), detailed logging, uses asyncio.Lock for state updates.
* **Functional Requirements Gaps:**
  * Missing periodic reconciliation with actual exchange balances/positions (FR-706).
  * Incomplete drawdown metrics (only total drawdown implemented, missing daily/weekly).
  * Basic position tracking (missing realized P\&L per position).
  * No explicit handling for CANCELED order events.
* **Design & Implementation Issues:**
  * Synchronous price retrieval (\_get\_latest\_price\_sync) called within async context, potentially blocking.
  * State updates occur in multiple places (handler, \_update\_portfolio\_value).
  * \_update\_portfolio\_value method has multiple responsibilities (calculation, state update, logging).
  * Potential inaccuracies if market price data for currency conversion is unavailable.
* **Configuration & Hardcoding:** Limited configurability (reconciliation frequency, drawdown resets), hardcoded defaults, global decimal precision setting.
* **Documentation Gaps:** Incomplete docstrings, lack of detail on financial calculations and state transitions.

## **2\. Whiteboard: Proposed Solutions**

Addressing the high and medium priority recommendations:

### **A. Implement Exchange Reconciliation (High Priority \- FR-706)**

* **Problem:** The internal state might drift from the actual exchange state over time due to missed events, API errors, or manual interventions.
* **Solution:**
  1. **Periodic Task:** Create an asyncio.Task (started in PortfolioManager.start) that runs periodically (interval configured via portfolio.reconciliation.interval\_seconds).
  2. **Fetch Exchange State:** This task needs to call methods (likely on the ExecutionHandler or a dedicated exchange adapter) to fetch current account balances and open positions directly from the Kraken API.
  3. **Compare & Log:** Compare the fetched exchange state with the internal \_available\_funds and \_positions. Log any discrepancies found, highlighting differences beyond a configurable threshold (portfolio.reconciliation.threshold).
  4. **Auto-Reconcile (Optional):** Add a configuration flag (portfolio.reconciliation.auto\_update). If true, automatically update the internal state (\_available\_funds, \_positions) to match the exchange state when discrepancies are found. *Caution:* Auto-reconciliation can mask underlying issues; manual review might be safer initially.
  5. **Locking:** Ensure reconciliation logic acquires the self.\_lock before modifying internal state variables.

\# In PortfolioManager class

\# Add in \_\_init\_\_:
\# self.\_reconciliation\_task: Optional\[asyncio.Task\] \= None
\# self.\_reconciliation\_interval \= self.config\_manager.get\_int("portfolio.reconciliation.interval\_seconds", 3600\) \# Default 1hr
\# self.\_reconciliation\_threshold \= Decimal(self.config\_manager.get("portfolio.reconciliation.threshold", "0.01")) \# Example threshold
\# self.\_auto\_reconcile \= self.config\_manager.get\_bool("portfolio.reconciliation.auto\_update", False)
\# \# Need access to execution handler or similar to fetch data
\# self.\_execution\_handler: Optional\[ExecutionHandler\] \= None \# Needs to be injected

async def start(self) \-\> None:
    \# ... (existing subscription logic) ...
    if self.\_reconciliation\_interval \> 0:
         self.logger.info(f"Starting periodic reconciliation every {self.\_reconciliation\_interval}s.", source\_module=self.\_\_class\_\_.\_\_name\_\_)
         self.\_reconciliation\_task \= asyncio.create\_task(self.\_run\_periodic\_reconciliation())
    else:
         self.logger.info("Exchange reconciliation disabled by configuration.", source\_module=self.\_\_class\_\_.\_\_name\_\_)

async def stop(self) \-\> None:
    \# ... (existing unsubscription logic) ...
    if self.\_reconciliation\_task and not self.\_reconciliation\_task.done():
         self.logger.info("Stopping reconciliation task...", source\_module=self.\_\_class\_\_.\_\_name\_\_)
         self.\_reconciliation\_task.cancel()
         try:
              await self.\_reconciliation\_task
         except asyncio.CancelledError:
              self.logger.info("Reconciliation task cancelled.", source\_module=self.\_\_class\_\_.\_\_name\_\_)
         except Exception as e:
              self.logger.error(f"Error stopping reconciliation task: {e}", exc\_info=True, source\_module=self.\_\_class\_\_.\_\_name\_\_)
         self.\_reconciliation\_task \= None

async def \_run\_periodic\_reconciliation(self) \-\> None:
    """Periodically reconciles internal state with the exchange."""
    while True:
        try:
            await asyncio.sleep(self.\_reconciliation\_interval)
            self.logger.info("Running periodic exchange reconciliation...", source\_module=self.\_\_class\_\_.\_\_name\_\_)
            await self.\_reconcile\_with\_exchange()
        except asyncio.CancelledError:
            self.logger.info("Reconciliation loop cancelled.", source\_module=self.\_\_class\_\_.\_\_name\_\_)
            break
        except Exception as e:
            self.logger.error(f"Error in reconciliation loop: {e}", exc\_info=True, source\_module=self.\_\_class\_\_.\_\_name\_\_)
            \# Avoid tight loop on error, wait before retrying
            await asyncio.sleep(self.\_reconciliation\_interval)

async def \_reconcile\_with\_exchange(self) \-\> None:
    """Fetches exchange state and compares/updates internal state."""
    if not self.\_execution\_handler or not hasattr(self.\_execution\_handler, 'get\_account\_balances') or not hasattr(self.\_execution\_handler, 'get\_open\_positions'):
        self.logger.warning("Execution handler not available or missing required methods for reconciliation.", source\_module=self.\_\_class\_\_.\_\_name\_\_)
        return

    try:
        \# Fetch data from exchange (needs methods on execution handler)
        exchange\_balances \= await self.\_execution\_handler.get\_account\_balances() \# Returns Dict\[str, Decimal\]
        exchange\_positions \= await self.\_execution\_handler.get\_open\_positions() \# Returns Dict\[str, PositionInfo\] or similar

        async with self.\_lock: \# Lock during comparison and potential update
            discrepancies\_found \= False

            \# \--- Reconcile Balances \---
            all\_currencies \= set(self.\_available\_funds.keys()) | set(exchange\_balances.keys())
            for currency in all\_currencies:
                internal\_bal \= self.\_available\_funds.get(currency, Decimal(0))
                exchange\_bal \= exchange\_balances.get(currency, Decimal(0))
                diff \= abs(internal\_bal \- exchange\_bal)

                if diff \> self.\_reconciliation\_threshold:
                    discrepancies\_found \= True
                    self.logger.warning(
                        f"Reconciliation: Balance mismatch for {currency}. "
                        f"Internal={internal\_bal:.8f}, Exchange={exchange\_bal:.8f}, Diff={diff:.8f}",
                        source\_module=self.\_\_class\_\_.\_\_name\_\_
                    )
                    if self.\_auto\_reconcile:
                        self.logger.info(f"Auto-reconciling {currency} balance to exchange value.", source\_module=self.\_\_class\_\_.\_\_name\_\_)
                        self.\_available\_funds\[currency\] \= exchange\_bal

            \# \--- Reconcile Positions \---
            all\_pairs \= set(self.\_positions.keys()) | set(exchange\_positions.keys())
            for pair in all\_pairs:
                 internal\_pos \= self.\_positions.get(pair)
                 exchange\_pos \= exchange\_positions.get(pair) \# Assuming exchange\_pos is PositionInfo or similar

                 internal\_qty \= internal\_pos.quantity if internal\_pos else Decimal(0)
                 exchange\_qty \= exchange\_pos.quantity if exchange\_pos else Decimal(0) \# Adapt based on actual return type
                 qty\_diff \= abs(internal\_qty \- exchange\_qty)

                 \# Add threshold check for quantity difference if needed
                 qty\_threshold \= Decimal("1e-8") \# Example small threshold
                 if qty\_diff \> qty\_threshold:
                      discrepancies\_found \= True
                      self.logger.warning(
                           f"Reconciliation: Position quantity mismatch for {pair}. "
                           f"Internal={internal\_qty:.8f}, Exchange={exchange\_qty:.8f}, Diff={qty\_diff:.8f}",
                           source\_module=self.\_\_class\_\_.\_\_name\_\_
                      )
                      if self.\_auto\_reconcile:
                           self.logger.info(f"Auto-reconciling {pair} position to exchange value.", source\_module=self.\_\_class\_\_.\_\_name\_\_)
                           if exchange\_pos:
                                \# Update or add position based on exchange data
                                \# This might need more sophisticated logic if avg price differs
                                self.\_positions\[pair\] \= exchange\_pos \# Simplistic update
                           elif internal\_pos:
                                \# Exchange shows no position, remove internal one
                                del self.\_positions\[pair\]

            if not discrepancies\_found:
                 self.logger.info("Reconciliation complete. No significant discrepancies found.", source\_module=self.\_\_class\_\_.\_\_name\_\_)
            else:
                 \# Recalculate portfolio value after potential reconciliation changes
                 await self.\_update\_portfolio\_value\_async() \# Use the async version

    except Exception as e:
        self.logger.error(f"Error during exchange reconciliation: {e}", exc\_info=True, source\_module=self.\_\_class\_\_.\_\_name\_\_)

### **B. Implement Daily/Weekly Drawdown Calculations (High Priority)**

* **Problem:** Only total drawdown (since inception or last peak) is calculated. Daily/Weekly limits are required by FR-503.
* **Solution:**
  1. **Add State:** Introduce \_daily\_peak\_equity, \_weekly\_peak\_equity, \_last\_daily\_reset\_time, \_last\_weekly\_reset\_time attributes in \_\_init\_\_. Initialize peaks to initial equity.
  2. **Modify \_update\_portfolio\_value (or a new \_update\_drawdown\_metrics):**
     * Get the current time (now \= datetime.utcnow()).
     * **Daily Reset:** Check if now is on a different day than \_last\_daily\_reset\_time. If so, reset \_daily\_peak\_equity to the *current* \_total\_equity and update \_last\_daily\_reset\_time.
     * **Weekly Reset:** Check if now is in a different week (e.g., check now.isocalendar().week or if now.weekday() is the reset day and it's a new week). If so, reset \_weekly\_peak\_equity and update \_last\_weekly\_reset\_time.
     * **Update Peaks:** If \_total\_equity \> \_daily\_peak\_equity, update \_daily\_peak\_equity. Same for weekly.
     * **Calculate Drawdowns:** Calculate \_daily\_drawdown\_pct and \_weekly\_drawdown\_pct based on their respective peak equities, similar to total drawdown calculation. Handle division by zero if peak equity is zero or negative.
  3. **Configuration:** Make reset times/days configurable (e.g., portfolio.drawdown.daily\_reset\_hour\_utc, portfolio.drawdown.weekly\_reset\_day).

\# In PortfolioManager class

\# Add in \_\_init\_\_:
\# self.\_daily\_peak\_equity: Decimal \= self.\_total\_equity
\# self.\_weekly\_peak\_equity: Decimal \= self.\_total\_equity
\# self.\_daily\_drawdown\_pct: Decimal \= Decimal(0)
\# self.\_weekly\_drawdown\_pct: Decimal \= Decimal(0)
\# self.\_last\_daily\_reset\_time: Optional\[datetime\] \= datetime.utcnow()
\# self.\_last\_weekly\_reset\_time: Optional\[datetime\] \= datetime.utcnow()
\# \# Load config for reset times
\# self.\_daily\_reset\_hour\_utc \= self.config\_manager.get\_int("portfolio.drawdown.daily\_reset\_hour\_utc", 0\)
\# self.\_weekly\_reset\_day \= self.config\_manager.get\_int("portfolio.drawdown.weekly\_reset\_day", 0\) \# 0=Monday

def \_update\_drawdown\_metrics(self) \-\> None:
    """Updates total, daily, and weekly drawdown metrics."""
    now \= datetime.utcnow()

    \# \--- Total Drawdown \---
    if self.\_total\_equity \> self.\_peak\_equity:
        self.\_peak\_equity \= self.\_total\_equity
    \# Calculate total drawdown % (ensure peak \> 0\)
    if self.\_peak\_equity \> Decimal(0):
        self.\_total\_drawdown\_pct \= ((self.\_peak\_equity \- self.\_total\_equity) / self.\_peak\_equity) \* 100
    else:
         self.\_total\_drawdown\_pct \= Decimal(0)

    \# \--- Daily Drawdown \---
    \# Check for reset condition (new day based on UTC reset hour)
    if self.\_last\_daily\_reset\_time is None or \\
       (now.date() \> self.\_last\_daily\_reset\_time.date()) or \\
       (now.date() \== self.\_last\_daily\_reset\_time.date() and now.hour \>= self.\_daily\_reset\_hour\_utc and self.\_last\_daily\_reset\_time.hour \< self.\_daily\_reset\_hour\_utc):
         self.logger.info(f"Resetting daily peak equity at {now}. Previous peak: {self.\_daily\_peak\_equity}", source\_module=self.\_\_class\_\_.\_\_name\_\_)
         self.\_daily\_peak\_equity \= self.\_total\_equity \# Reset to current equity
         self.\_last\_daily\_reset\_time \= now
         self.\_daily\_drawdown\_pct \= Decimal(0) \# Reset drawdown too

    \# Update daily peak
    if self.\_total\_equity \> self.\_daily\_peak\_equity:
        self.\_daily\_peak\_equity \= self.\_total\_equity
    \# Calculate daily drawdown %
    if self.\_daily\_peak\_equity \> Decimal(0):
        self.\_daily\_drawdown\_pct \= ((self.\_daily\_peak\_equity \- self.\_total\_equity) / self.\_daily\_peak\_equity) \* 100
    else:
         self.\_daily\_drawdown\_pct \= Decimal(0)

    \# \--- Weekly Drawdown \---
    \# Check for reset condition (new week, reset on configured day)
    is\_reset\_day \= (now.weekday() \== self.\_weekly\_reset\_day)
    is\_new\_week \= (self.\_last\_weekly\_reset\_time is None or now.isocalendar().week \!= self.\_last\_weekly\_reset\_time.isocalendar().week or now.year \!= self.\_last\_weekly\_reset\_time.year)

    if is\_reset\_day and is\_new\_week:
        self.logger.info(f"Resetting weekly peak equity at {now}. Previous peak: {self.\_weekly\_peak\_equity}", source\_module=self.\_\_class\_\_.\_\_name\_\_)
        self.\_weekly\_peak\_equity \= self.\_total\_equity \# Reset to current equity
        self.\_last\_weekly\_reset\_time \= now
        self.\_weekly\_drawdown\_pct \= Decimal(0) \# Reset drawdown

    \# Update weekly peak
    if self.\_total\_equity \> self.\_weekly\_peak\_equity:
        self.\_weekly\_peak\_equity \= self.\_total\_equity
    \# Calculate weekly drawdown %
    if self.\_weekly\_peak\_equity \> Decimal(0):
        self.\_weekly\_drawdown\_pct \= ((self.\_weekly\_peak\_equity \- self.\_total\_equity) / self.\_weekly\_peak\_equity) \* 100
    else:
         self.\_weekly\_drawdown\_pct \= Decimal(0)

    self.logger.debug(
         f"Drawdown Update: Total={self.\_total\_drawdown\_pct:.2f}%, "
         f"Daily={self.\_daily\_drawdown\_pct:.2f}% (Peak={self.\_daily\_peak\_equity:.4f}), "
         f"Weekly={self.\_weekly\_drawdown\_pct:.2f}% (Peak={self.\_weekly\_peak\_equity:.4f})",
         source\_module=self.\_\_class\_\_.\_\_name\_\_
    )

\# Ensure \_update\_drawdown\_metrics is called within \_update\_portfolio\_value\_async
async def \_update\_portfolio\_value\_async(self) \-\> None:
    \# ... (calculate cash\_value, position\_value) ...
    current\_total\_value \= cash\_value \+ position\_value
    \# ... (handle missing prices) ...
    async with self.\_lock: \# Lock before updating state
         self.\_total\_equity \= current\_total\_value
         self.\_update\_drawdown\_metrics() \# Call drawdown update here
         \# Log combined state if needed
         self.\_log\_updated\_state() \# Ensure this logs the new drawdown values too

### **C. Redesign Price Retrieval (High Priority)**

* **Problem:** \_get\_latest\_price\_sync calls an async method (market\_price\_service.get\_latest\_price) synchronously, which is incorrect and likely blocks the event loop or fails. The get\_current\_state method needs prices synchronously, but the update mechanism (\_handle\_execution\_report \-\> \_update\_portfolio\_value) is async.
* **Solution:**
  1. **Remove \_get\_latest\_price\_sync:** Eliminate this problematic synchronous wrapper.
  2. **Asynchronous Price Fetching:** All interactions with MarketPriceService **must** use await.
  3. **Refactor \_update\_portfolio\_value:** Make this method fully async (\_update\_portfolio\_value\_async) as it needs to await price fetching for valuation. Call this async version from \_handle\_execution\_report.
  4. **Refactor get\_current\_state:** This method *must* remain synchronous according to the interface definition (Section 4.1 of inter-module comm doc) because RiskManager needs immediate, consistent state.
     * **Option 1 (Stale Price OK):** get\_current\_state returns the state based on the *last known* prices calculated asynchronously by \_update\_portfolio\_value\_async. It does *not* fetch live prices itself. This is simpler but means the state might be slightly stale between updates. Add the timestamp of the last async update to the returned state.
     * **Option 2 (Dedicated Price Cache):** Maintain a simple internal dictionary cache (\_latest\_prices: Dict\[str, Tuple\[Decimal, datetime\]\]) that is updated *asynchronously* whenever new price data is available (e.g., by subscribing PortfolioManager to MarketDataL2Event or a dedicated PriceUpdateEvent). get\_current\_state reads synchronously from this cache. This provides fresher prices but adds complexity.
  * **Recommendation:** Start with Option 1 for simplicity. The state is updated after every execution report, which might be frequent enough for risk checks. Document the potential staleness. If higher price freshness is needed for get\_current\_state, implement Option 2 later.

\# In PortfolioManager class

\# Remove \_get\_latest\_price\_sync

\# Ensure \_update\_portfolio\_value\_async uses await:
async def \_calculate\_single\_position\_value(
    self, pair: str, position: PositionInfo
) \-\> Tuple\[Decimal, bool\]:
    \# \--- Use await \---
    market\_price \= await self.market\_price\_service.get\_latest\_price(pair)
    \# ... rest of calculation ...

async def \_convert\_currency\_value(self, currency: str, amount: Decimal) \-\> Tuple\[Decimal, bool\]:
    \# ... construct pairs ...
    \# \--- Use await \---
    conversion\_rate \= await self.market\_price\_service.get\_latest\_price(conversion\_pair)
    \# ... rest of calculation using await for inverse price too ...

\# Ensure \_handle\_execution\_report calls the async version
async def \_handle\_execution\_report(self, event: "ExecutionReportEvent") \-\> None:
    \# ... (parsing and initial updates) ...
    async with self.\_lock:
        \# ... (update funds, position, commission) ...
        \# \--- Call async version \---
        await self.\_update\_portfolio\_value\_async()
        \# Logging is now handled inside \_update\_portfolio\_value\_async or called after await

\# Modify get\_current\_state (Option 1: Use last calculated values)
def get\_current\_state(self) \-\> Dict\[str, Any\]:
    """\*\*Synchronous Method\*\*
    Returns the portfolio state based on the last asynchronous update.
    Does NOT fetch live prices. Prices reflect state after last execution report.
    """
    self.logger.debug("get\_current\_state called (using last calculated values).", source\_module=self.\_\_class\_\_.\_\_name\_\_)
    \# Acquire lock for reading state consistency? Depends on whether async updates
    \# could happen concurrently with this read in a single-threaded asyncio model.
    \# If PortfolioManager runs solely within the main asyncio thread, lock might
    \# not be strictly needed for reads, but doesn't hurt.
    \# async with self.\_lock: \# Cannot use async lock in sync method\!
    \# If strict consistency needed, might need thread lock or redesign.
    \# Assuming reads are safe enough without lock in typical asyncio usage:

    positions\_dict \= {}
    \# Need last known prices \- store them when \_update\_portfolio\_value\_async runs
    \# Add self.\_last\_known\_prices: Dict\[str, Decimal\] \= {}
    \# Add self.\_last\_state\_update\_time: Optional\[datetime\] \= None

    for pair, pos\_info in self.\_positions.items():
        if abs(pos\_info.quantity) \> Decimal("1e-12"): \# Check threshold
            latest\_price \= self.\_last\_known\_prices.get(pair) \# Read from cache
            \# ... calculate market\_value, unrealized\_pnl using cached price ...
            positions\_dict\[pair\] \= {
                 \# ... include 'current\_market\_value', 'unrealized\_pnl' based on cached price ...
                 "quantity": str(pos\_info.quantity),
                 "average\_entry\_price": str(pos\_info.average\_entry\_price),
                 \# ...
            }

    \# ... (calculate exposure based on cached prices/values) ...

    return {
        "timestamp": self.\_last\_state\_update\_time.isoformat() \+ "Z" if self.\_last\_state\_update\_time else None, \# Add timestamp of last update
        "valuation\_currency": self.valuation\_currency,
        "total\_equity": str(self.\_total\_equity), \# Last calculated value
        "available\_funds": {k: str(v) for k, v in self.\_available\_funds.items()},
        "positions": positions\_dict,
        "total\_exposure\_pct": str(self.\_last\_total\_exposure\_pct), \# Need to cache this too
        "daily\_drawdown\_pct": str(self.\_daily\_drawdown\_pct), \# Last calculated
        "weekly\_drawdown\_pct": str(self.\_weekly\_drawdown\_pct), \# Last calculated
        "total\_drawdown\_pct": str(self.\_total\_drawdown\_pct), \# Last calculated
    }

\# Modify \_update\_portfolio\_value\_async to cache prices/exposure
async def \_update\_portfolio\_value\_async(self) \-\> None:
     \# ... perform async price fetches and calculations ...
     async with self.\_lock:
          self.\_total\_equity \= current\_total\_value
          self.\_update\_drawdown\_metrics()
          \# \--- Cache values needed by get\_current\_state \---
          self.\_last\_known\_prices \= calculated\_prices\_dict \# Store prices used
          self.\_last\_total\_exposure\_pct \= calculated\_exposure\_pct \# Store exposure
          self.\_last\_state\_update\_time \= datetime.utcnow()
          \# \--- End Caching \---
          self.\_log\_updated\_state()

### **D. Improve Configuration Management (Medium Priority)**

* **Problem:** Hardcoded defaults, global decimal precision setting.
* **Solution:**
  1. **Load All Config:** Load all necessary parameters (reconciliation interval/threshold/auto, drawdown reset times, decimal precision) from ConfigManager in \_\_init\_\_, providing defaults only within the config\_manager.get call.
  2. **Decimal Context:** Avoid setting global precision. If specific precision is needed for internal calculations, create a local decimal.Context in \_\_init\_\_ based on config and use it explicitly in calculations (with local\_context: result \= a \+ b) or apply it when formatting outputs (value.quantize(Decimal('0.01'), context=local\_context)). Standard Decimal operations will use the default precision unless overridden.

### **E. Add Realized P\&L Tracking & Cancellation Handling (Medium Priority)**

* **Problem:** Realized P\&L isn't tracked per position; cancelled orders aren't explicitly handled.
* **Solution:**
  1. **Realized P\&L:** When processing a closing fill (SELL for a long position, BUY for a short) in \_handle\_execution\_report, calculate the realized P\&L for the filled portion ((exit\_price \- avg\_entry\_price) \* quantity\_filled adjusted for side). Store this, perhaps in a separate list/log or aggregate it within the PositionInfo dataclass (add realized\_pnl field).
  2. **Cancellation Handling:** Modify \_validate\_execution\_report to recognize order\_status \== "CANCELED". Log the cancellation. If tracking pending orders internally (which isn't shown currently but might be needed), update the internal state of that order to cancelled. It generally won't affect cash or position quantity unless it was a partially filled order being cancelled.

**Conclusion:** Implementing exchange reconciliation and correct drawdown calculations are key functional requirements. Fixing the synchronous price fetching logic is critical for correctness within the asyncio framework. Enhancing configuration and adding more detailed P\&L tracking will improve robustness and utility.
