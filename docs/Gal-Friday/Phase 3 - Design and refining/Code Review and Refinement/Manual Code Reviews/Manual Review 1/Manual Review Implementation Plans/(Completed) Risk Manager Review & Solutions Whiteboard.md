# **Risk Manager (risk\_manager.py): Review Summary & Solution Whiteboard**

## **1\. Summary of Manual Review Findings (risk\_manager.md)**

* **Strengths:** Comprehensive risk configuration loading, correct fixed fractional position sizing implementation, proper stop-loss validation (including distance), effective periodic drawdown monitoring and HALT triggering, correct use of Decimal.
* **Functional Requirements Gaps (High Priority):**
  * Missing "fat finger" check for entry prices against current market price (FR-506).
  * Missing pre-trade checks for maximum total portfolio exposure and sufficient balance (FR-506).
  * Missing tracking and enforcement of the consecutive losing trades limit (FR-504).
  * Limited validation/handling of Take Profit (TP) prices beyond generating a default.
* **Design & Implementation Issues:**
  * Uses TYPE\_CHECKING but lacks runtime import fallback for PortfolioManager (unlike other modules reviewed, potentially causing runtime errors if dependencies aren't perfect).
  * Sets global Decimal precision, which can affect other modules unexpectedly.
  * Defines unused TradeSignalProposedPayload and SystemHaltPayload dataclasses.
  * Inconsistent error handling (returning tuples vs. raising exceptions).
* **Error Handling Concerns:** Limited recovery if PortfolioManager.get\_current\_state() fails; doesn't check position size against exchange min/max limits. No validation of risk configuration values.
* **Configuration & Hardcoding:** Some default risk limits might be too aggressive; default TP calculation uses a hardcoded 2x risk multiplier.

## **2\. Whiteboard: Proposed Solutions**

Addressing the high and medium priority recommendations:

### **A. Implement Missing Pre-Trade Checks (High Priority \- FR-506)**

* **Problem:** Key pre-trade checks for exposure, balance, and price sanity ("fat finger") are missing.
* **Solution:** Add these checks within the \_perform\_pre\_trade\_checks method *before* approving the signal.
  1. **Fat Finger Check:**
     * Requires access to the current market price (e.g., via MarketPriceService injected into RiskManager or passed through from StrategyArbitrator within the TradeSignalProposedEvent).
     * Compare the proposed\_entry\_price (or reference price) to the current market price.
     * Reject if the deviation exceeds a configurable percentage (risk\_manager.max\_price\_deviation\_pct).
  2. **Portfolio Exposure Check:**
     * Get total\_exposure\_pct and total\_equity from portfolio\_state.
     * Calculate the value of the *proposed* trade (calculated\_qty \* entry\_price). Convert this value to the portfolio's valuation currency if necessary (requires price conversion logic, potentially using MarketPriceService).
     * Calculate the *new* total exposure percentage if the trade were added.
     * Reject if new\_exposure\_pct exceeds \_max\_total\_exposure\_pct.
  3. **Sufficient Balance Check:**
     * Determine the quote currency needed for the trade (e.g., USD for XRP/USD BUY).
     * Calculate the estimated cost (calculated\_qty \* entry\_price \+ estimated fees).
     * Get the available\_funds for the quote currency from portfolio\_state.
     * Reject if available\_funds \< estimated\_cost.

\# In RiskManager class

\# Add MarketPriceService to \_\_init\_\_ if needed for fat finger check
\# def \_\_init\_\_(..., market\_price\_service: "MarketPriceService"):
\#    ...
\#    self.market\_price\_service \= market\_price\_service
\#    ...

async def \_perform\_pre\_trade\_checks( \# Make async if price fetching is needed
    self, event: TradeSignalProposedEvent
) \-\> Tuple\[bool, Optional\[str\], Optional\[Dict\[str, Any\]\]\]:
    \# ... (get portfolio\_state, state\_values, check drawdowns as before) ...
    \# ... (validate entry\_price, sl\_price as before) ...

    \# \--- Fat Finger Check \---
    \# Requires current market price \- needs MarketPriceService or similar
    \# Assuming self.market\_price\_service exists and is async
    current\_market\_price \= await self.market\_price\_service.get\_latest\_price(event.trading\_pair)
    if current\_market\_price is None:
         return False, "MARKET\_PRICE\_UNAVAILABLE\_FOR\_FAT\_FINGER", None

    max\_deviation\_pct \= self.\_config.get\_decimal("fat\_finger\_max\_deviation\_pct", Decimal("5.0")) \# Example config
    deviation\_pct \= abs(entry\_price \- current\_market\_price) / current\_market\_price \* 100 if current\_market\_price \> 0 else Decimal('Infinity')

    if deviation\_pct \> max\_deviation\_pct:
         reason \= f"FAT\_FINGER\_CHECK\_FAILED ({deviation\_pct:.2f}% \> {max\_deviation\_pct}%)"
         self.logger.warning(f"Signal {signal\_id} rejected: {reason}", source\_module=self.\_source\_module)
         return False, reason, None
    \# \--- End Fat Finger Check \---

    \# Calculate position size (as before)
    calculated\_qty \= self.\_calculate\_position\_size(...)
    if calculated\_qty is None or calculated\_qty \<= 0:
        return False, "POSITION\_SIZE\_CALCULATION\_FAILED", None

    \# \--- Check Position Size vs Exchange Limits \--- \#
    \# Needs access to pair info (min/max order size) \- potentially from ConfigManager or a dedicated service
    \# min\_order\_size \= self.\_get\_min\_order\_size(event.trading\_pair)
    \# max\_order\_size \= self.\_get\_max\_order\_size(event.trading\_pair)
    \# if calculated\_qty \< min\_order\_size: return False, f"QTY\_BELOW\_MIN ({calculated\_qty} \< {min\_order\_size})", None
    \# if calculated\_qty \> max\_order\_size: return False, f"QTY\_ABOVE\_MAX ({calculated\_qty} \> {max\_order\_size})", None
    \# \--- End Exchange Limit Check \--- \#

    \# \--- Portfolio Exposure Check \---
    \# Calculate value of this trade in valuation currency
    base\_asset, quote\_asset \= self.\_split\_symbol(event.trading\_pair) \# Need helper
    trade\_value\_quote \= calculated\_qty \* entry\_price
    trade\_value\_valuation\_ccy, conversion\_error \= await self.\_convert\_to\_valuation\_ccy(trade\_value\_quote, quote\_asset)

    if conversion\_error:
         return False, f"CURRENCY\_CONVERSION\_FAILED\_FOR\_EXPOSURE ({conversion\_error})", None

    current\_exposure\_pct \= Decimal(portfolio\_state.get("total\_exposure\_pct", "0"))
    equity \= state\_values\["current\_equity"\]
    new\_exposure\_increment\_pct \= (abs(trade\_value\_valuation\_ccy) / equity) \* 100 if equity \> 0 else Decimal('Infinity')
    new\_total\_exposure\_pct \= current\_exposure\_pct \+ new\_exposure\_increment\_pct

    if new\_total\_exposure\_pct \> self.\_max\_total\_exposure\_pct:
        reason \= f"MAX\_TOTAL\_EXPOSURE\_LIMIT ({new\_total\_exposure\_pct:.2f}% \> {self.\_max\_total\_exposure\_pct}%)"
        self.logger.warning(f"Signal {signal\_id} rejected: {reason}", source\_module=self.\_source\_module)
        return False, reason, None
    \# \--- End Portfolio Exposure Check \---

    \# \--- Sufficient Balance Check \---
    \# Estimate cost (including fees \- need fee rate from config)
    fee\_rate \= self.\_config.get\_decimal("exchange.taker\_fee\_pct", Decimal("0.26")) / 100 \# Example config
    estimated\_cost \= trade\_value\_quote \* (1 \+ fee\_rate) if event.side \== "BUY" else Decimal(0) \# Cost only for BUYs usually
    \# For SELLs, check if base asset balance is sufficient? More complex. Assume check BUY cost for now.

    if event.side \== "BUY":
         available\_quote\_funds \= Decimal(portfolio\_state.get("available\_funds", {}).get(quote\_asset, "0"))
         if available\_quote\_funds \< estimated\_cost:
              reason \= f"INSUFFICIENT\_FUNDS ({quote\_asset}: {available\_quote\_funds:.4f} \< {estimated\_cost:.4f})"
              self.logger.warning(f"Signal {signal\_id} rejected: {reason}", source\_module=self.\_source\_module)
              return False, reason, None
    \# \--- End Sufficient Balance Check \---

    \# \--- Check Consecutive Losses \---
    loss\_check\_ok, loss\_reason \= self.\_check\_consecutive\_losses() \# See Section B
    if not loss\_check\_ok:
         return False, loss\_reason, None
    \# \--- End Consecutive Losses Check \---

    \# All checks passed
    self.logger.info(f"Signal {signal\_id} passed all pre-trade checks.", source\_module=self.\_source\_module)
    \# ... (prepare approved\_payload as before) ...
    return True, None, approved\_payload

\# \--- Helper Methods Needed \---
\# async def \_get\_current\_market\_price(self, trading\_pair: str) \-\> Optional\[Decimal\]: ... \# Uses self.market\_price\_service
\# def \_split\_symbol(self, symbol: str) \-\> Tuple\[str, str\]: ... \# Already in PortfolioManager, maybe move to utils?
\# async def \_convert\_to\_valuation\_ccy(self, amount: Decimal, currency: str) \-\> Tuple\[Optional\[Decimal\], Optional\[str\]\]: ... \# Needs price conversion logic
\# def \_get\_min\_order\_size(...) / \_get\_max\_order\_size(...): ... \# Needs exchange info access

### **B. Implement Consecutive Losses Tracking (High Priority \- FR-504)**

* **Problem:** The system doesn't track or limit consecutive losing trades.
* **Solution:**
  1. **Add State:** Add \_consecutive\_loss\_count: int \= 0 to \_\_init\_\_.
  2. **Subscribe:** Subscribe RiskManager to ExecutionReportEvent (specifically interested in filled orders that close a trade).
  3. **Update Count:** In the handler (\_handle\_execution\_report\_for\_losses):
     * Identify if the fill closes a trade (requires linking fills back to signals/trades, potentially state from PortfolioManager).
     * Determine if the closed trade was a loss (requires P\&L calculation, potentially from PortfolioManager state or the event itself if added).
     * If loss, increment \_consecutive\_loss\_count.
     * If profit, reset \_consecutive\_loss\_count \= 0\.
  4. **Check Limit:** Implement \_check\_consecutive\_losses() method called during pre-trade checks (see Section A). This method compares \_consecutive\_loss\_count to \_max\_consecutive\_losses from config. If the limit is reached, it returns False and a reason string.

\# In RiskManager class

\# Add in \_\_init\_\_:
\# self.\_consecutive\_loss\_count: int \= 0
\# self.\_exec\_report\_handler \= self.\_handle\_execution\_report\_for\_losses \# Store for unsubscribe

\# Add in start():
\# await self.pubsub.subscribe(EventType.EXECUTION\_REPORT, self.\_exec\_report\_handler)

\# Add in stop():
\# await self.pubsub.unsubscribe(EventType.EXECUTION\_REPORT, self.\_exec\_report\_handler)

async def \_handle\_execution\_report\_for\_losses(self, event: ExecutionReportEvent):
    \# Simplified logic: Assume event indicates a closed trade and has PnL
    \# A real implementation needs robust trade tracking from PortfolioManager
    if event.order\_status \== "FILLED": \# Or based on specific trade closure logic
         \# \--- Determine PnL \---
         \# This is complex. PortfolioManager should ideally publish a
         \# 'TradeClosedEvent' with PnL, or RiskManager needs to query PM.
         \# Placeholder: Assume event has 'realized\_pnl' attribute
         realized\_pnl \= getattr(event, 'realized\_pnl', None) \# Needs PnL on event
         if realized\_pnl is None:
              \# Cannot determine PnL from this event alone
              return

         if isinstance(realized\_pnl, (str, float)): \# Convert if needed
              realized\_pnl \= Decimal(str(realized\_pnl))

         \# \--- Update Counter \---
         if realized\_pnl \< 0:
              self.\_consecutive\_loss\_count \+= 1
              self.logger.info(f"Consecutive loss count increased to {self.\_consecutive\_loss\_count}", source\_module=self.\_source\_module)
         elif realized\_pnl \> 0: \# Reset on profit
              if self.\_consecutive\_loss\_count \> 0:
                   self.logger.info(f"Consecutive loss streak broken. Resetting count from {self.\_consecutive\_loss\_count}.", source\_module=self.\_source\_module)
                   self.\_consecutive\_loss\_count \= 0
         \# else: Zero PnL, count doesn't change

         \# \--- Check Limit Immediately (Optional \- could also be done only pre-trade) \---
         if self.\_consecutive\_loss\_count \>= self.\_max\_consecutive\_losses:
              reason \= f"Consecutive loss limit reached: {self.\_consecutive\_loss\_count} \>= {self.\_max\_consecutive\_losses}"
              self.logger.critical(f"HALT Trigger Condition: {reason}", source\_module=self.\_source\_module)
              \# Publish PotentialHaltTriggerEvent
              halt\_event \= PotentialHaltTriggerEvent(
                   source\_module=self.\_source\_module, event\_id=uuid.uuid4(),
                   timestamp=datetime.utcnow(), reason=reason
              )
              await self.pubsub.publish(halt\_event)

def \_check\_consecutive\_losses(self) \-\> Tuple\[bool, Optional\[str\]\]:
    """Checks if the consecutive loss limit has been reached."""
    if self.\_consecutive\_loss\_count \>= self.\_max\_consecutive\_losses:
        reason \= f"MAX\_CONSECUTIVE\_LOSSES\_LIMIT ({self.\_consecutive\_loss\_count} \>= {self.\_max\_consecutive\_losses})"
        self.logger.warning(f"Trade rejected: {reason}", source\_module=self.\_source\_module)
        return False, reason
    return True, None

### **C. Fix Decimal Precision & TYPE\_CHECKING (Medium Priority)**

* **Problem:** Global decimal precision set; TYPE\_CHECKING used without runtime fallback import for PortfolioManager.
* **Solution:**
  1. **Decimal Precision:** Remove getcontext().prec \= 28\. If specific precision is needed for calculations within RiskManager, create and use a local decimal.Context as shown in the portfolio\_manager.md recommendations. Ensure Decimal objects passed between modules maintain sufficient precision.
  2. **TYPE\_CHECKING:** Add a runtime import block for PortfolioManager similar to how it's handled in other modules reviewed (like kraken.py or monitoring\_service.py) to prevent NameError if type checking is off.

\# In risk\_manager.py

\# Remove: getcontext().prec \= 28

\# Add runtime import block:
if TYPE\_CHECKING:
    from .portfolio\_manager import PortfolioManager
else:
    \# Define placeholder or attempt runtime import carefully
    try:
        from .portfolio\_manager import PortfolioManager
    except ImportError:
        \# Define a minimal placeholder if import fails at runtime
        \# This allows basic script execution but will fail if methods are called
        class PortfolioManager: \# type: ignore
             def get\_current\_state(self) \-\> Dict\[str, Any\]: return {}
             \# Add other methods used by RiskManager if necessary

\# If using local context for precision:
\# def \_\_init\_\_(...):
\#     ...
\#     self.\_decimal\_context \= getcontext().copy()
\#     self.\_decimal\_context.prec \= self.\_config.get\_int("risk\_manager.decimal\_precision", 28\)
\#
\# def \_calculate\_position\_size(...):
\#     with self.\_decimal\_context:
\#         \# Perform calculations needing specific precision
\#         risk\_amount\_quote \= current\_equity \* (risk\_per\_trade\_pct / 100\)
\#         ...

### **D. Improve Portfolio State Error Handling (Medium Priority)**

* **Problem:** Failure to get portfolio state skips the check cycle without robust recovery.
* **Solution:** Implement a retry mechanism (\_get\_portfolio\_state\_with\_retry) when calling \_portfolio\_manager.get\_current\_state(). Use a simple loop with a short sleep. If state remains unavailable after retries, reject the trade signal with a specific reason.
  \# In RiskManager class

  \# Replace direct call in \_perform\_pre\_trade\_checks:
  \# portfolio\_state \= self.\_get\_portfolio\_state()
  \# With:
  portfolio\_state \= self.\_get\_portfolio\_state\_with\_retry() \# Use retry wrapper

  \# Add the retry wrapper method:
  def \_get\_portfolio\_state\_with\_retry(self, max\_retries: int \= 2, retry\_delay\_s: float \= 0.1) \-\> Optional\[Dict\[str, Any\]\]:
       """Attempts to get portfolio state with simple retries."""
       for attempt in range(max\_retries \+ 1):
            try:
                 state \= self.\_portfolio\_manager.get\_current\_state()
                 if state: \# Check if state is not None or empty
                      return state
                 \# Log warning if state is None/empty even without exception
                 self.logger.warning(f"PortfolioManager returned empty/None state (Attempt {attempt+1})", source\_module=self.\_source\_module)
            except Exception as e:
                 self.logger.warning(
                      f"Error getting portfolio state (Attempt {attempt+1}/{max\_retries+1}): {e}",
                      source\_module=self.\_source\_module
                 )
            \# Wait before retrying (use time.sleep as this method is synchronous)
            if attempt \< max\_retries:
                 import time
                 time.sleep(retry\_delay\_s)

       self.logger.error(f"Failed to get valid portfolio state after {max\_retries+1} attempts.", source\_module=self.\_source\_module)
       return None

### **E. Improve Configuration & Hardcoding (Medium Priority)**

* **Problem:** Hardcoded defaults for risk limits and TP multiplier. No config validation.
* **Solution:**
  1. **Defaults:** Ensure *all* risk parameters loaded in \_load\_config fetch defaults using config\_manager.get(...) rather than hardcoding them in the get call's default argument if possible, or at least log warnings when defaults are used.
  2. **TP Multiplier:** Add a configuration parameter risk\_manager.sizing.default\_tp\_rr\_ratio (defaulting to 2.0) and use it in \_publish\_trade\_signal\_approved instead of the hardcoded \* 2\.
  3. **Config Validation:** Implement a \_validate\_config method called after \_load\_config. Check that critical values are present and within reasonable bounds (e.g., percentages between 0-100). Log errors or raise exceptions if validation fails.

**Conclusion:** The RiskManager is functional but needs the missing pre-trade checks (fat finger, exposure, balance, consecutive losses) implemented to meet requirements (FR-504, FR-506). Addressing the decimal precision setting, improving error handling for portfolio state retrieval, and enhancing configurability will significantly increase its robustness and maintainability.
