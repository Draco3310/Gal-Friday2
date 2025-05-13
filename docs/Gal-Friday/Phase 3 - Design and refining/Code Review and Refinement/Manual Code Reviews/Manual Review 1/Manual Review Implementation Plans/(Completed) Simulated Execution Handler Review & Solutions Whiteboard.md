# **Simulated Execution Handler (simulated\_execution\_handler.py): Review Summary & Solution Whiteboard**

## **1\. Summary of Manual Review Findings (simulated\_execution\_handler.md)**

* **Strengths:** Interface compatible with real handler, simulates market & limit orders, configurable fees/slippage, realistic limit order fill check, publishes correct ExecutionReportEvent.
* **Functional Requirements Gaps (High Priority):**
  * Missing Stop-Loss/Take-Profit (SL/TP) order simulation (FR-606). The handler processes the entry order but doesn't simulate the subsequent SL/TP execution based on price movement.
  * No Partial Fills simulation (FR-607). Assumes orders fill entirely or not at all.
  * Limited Order Types: Only simulates Market and Limit; Stop orders aren't handled.
  * No Limit Order Timeout simulation (FR-605). Limit orders that don't fill immediately are simply rejected in the next bar check, without a configurable timeout period.
* **Design & Implementation Issues:** Inconsistent error handling, potentially inefficient data retrieval (\_get\_next\_bar\_data doesn't specify range), hard dependency on OHLCV bar structure.
* **Simulation Quality Concerns:** Basic slippage models (fixed %, ATR), no simulation of latency, market gaps, or exchange-specific quirks.
* **Error Handling/Validation:** Limited validation of incoming order parameters (e.g., size vs. exchange limits).

## **2\. Whiteboard: Proposed Solutions**

Addressing the high and medium priority recommendations:

### **A. Implement SL/TP Order Simulation (High Priority \- FR-606)**

* **Problem:** After simulating an entry fill, the corresponding SL/TP orders defined in the TradeSignalApprovedEvent are not simulated.
* **Solution:**
  1. **Trigger:** After successfully simulating an entry fill (within handle\_trade\_signal\_approved or upon receiving the "FILLED" report for the entry), store the active SL/TP prices associated with that position/signal ID.
  2. **Monitoring:** In the simulation loop (or triggered by subsequent bar data arrival):
     * Retrieve the next historical bar(s) after the entry fill timestamp.
     * Check if the high or low of these subsequent bars crosses the stored SL or TP price levels.
     * The first level breached determines the exit. SL is triggered if low \<= sl\_price (for longs) or high \>= sl\_price (for shorts). TP is triggered if high \>= tp\_price (for longs) or low \<= tp\_price (for shorts).
  3. **Execution:** If an SL or TP is triggered:
     * Determine the fill price (e.g., the SL/TP price itself, or the bar's open/close after the breach, potentially with slippage).
     * Publish an ExecutionReportEvent for the simulated SL/TP fill (opposite side, same quantity as entry). Mark the original position as closed.
  4. **Complexity:** Requires managing the state of "active" SL/TP levels linked to open simulated positions.

\# In SimulatedExecutionHandler class

\# Add state to track active SL/TP levels (e.g., in \_\_init\_\_)
\# self.\_active\_sl\_tp: Dict\[str, Dict\[str, Optional\[Decimal\]\]\] \= defaultdict(dict) \# signal\_id \-\> {"sl": Decimal, "tp": Decimal}

async def handle\_trade\_signal\_approved(self, event: "TradeSignalApprovedEvent") \-\> None:
    \# ... (simulate entry fill logic) ...
    if fill\_result and fill\_result\["status"\] in \["FILLED", "PARTIALLY\_FILLED"\]:
        \# Store SL/TP for the filled quantity (handle partial fills later)
        \# Use a unique identifier for the trade/position, e.g., event.signal\_id or client\_order\_id
        position\_id \= str(event.signal\_id) \# Example identifier
        self.\_active\_sl\_tp\[position\_id\] \= {
            "sl": event.sl\_price,
            "tp": event.tp\_price,
            "side": event.side, \# Store original side
            "pair": event.trading\_pair,
            "entry\_qty": fill\_result\["quantity"\], \# Store filled quantity
            "entry\_ts": fill\_result\["timestamp"\], \# Store fill time
            "entry\_event": event \# Store original event for reporting
        }
        self.logger.info(f"Stored SL/TP for position {position\_id}: SL={event.sl\_price}, TP={event.tp\_price}", source\_module=self.\_\_class\_\_.\_\_name\_\_)
        \# \--- Trigger SL/TP monitoring for this position \---
        \# This needs integration with the backtesting loop's time progression.
        \# The backtester needs to call a method like \`check\_active\_sl\_tp(current\_bar)\`
        \# for each new bar after the entry.

async def check\_active\_sl\_tp(self, current\_bar: pd.Series, bar\_timestamp: datetime) \-\> None:
    """Called by the backtesting engine for each new bar to check SL/TP triggers."""
    if not hasattr(current\_bar, 'high') or not hasattr(current\_bar, 'low'):
         self.logger.warning("Current bar missing high/low data for SL/TP check.", source\_module=self.\_\_class\_\_.\_\_name\_\_)
         return

    bar\_high \= Decimal(str(current\_bar\["high"\]))
    bar\_low \= Decimal(str(current\_bar\["low"\]))
    \# Use bar\_timestamp which should be the start time of the current\_bar

    triggered\_positions \= \[\]
    for position\_id, sl\_tp\_data in list(self.\_active\_sl\_tp.items()): \# Iterate over copy
        if bar\_timestamp \<= sl\_tp\_data\["entry\_ts"\]: \# Don't check bars before/at entry
             continue

        sl\_price \= sl\_tp\_data\["sl"\]
        tp\_price \= sl\_tp\_data\["tp"\]
        side \= sl\_tp\_data\["side"\]
        pair \= sl\_tp\_data\["pair"\]
        entry\_qty \= sl\_tp\_data\["entry\_qty"\]
        originating\_event \= sl\_tp\_data\["entry\_event"\]

        exit\_side \= "SELL" if side \== "BUY" else "BUY"
        exit\_price \= None
        exit\_reason \= None

        \# \--- Check SL \---
        if sl\_price:
            if side \== "BUY" and bar\_low \<= sl\_price:
                exit\_price \= sl\_price \# Simulate fill at stop price
                exit\_reason \= "Stop Loss triggered"
            elif side \== "SELL" and bar\_high \>= sl\_price:
                exit\_price \= sl\_price
                exit\_reason \= "Stop Loss triggered"

        \# \--- Check TP (only if SL not already triggered) \---
        if exit\_price is None and tp\_price:
             if side \== "BUY" and bar\_high \>= tp\_price:
                  exit\_price \= tp\_price \# Simulate fill at limit price
                  exit\_reason \= "Take Profit triggered"
             elif side \== "SELL" and bar\_low \<= tp\_price:
                  exit\_price \= tp\_price
                  exit\_reason \= "Take Profit triggered"

        \# \--- If Triggered \---
        if exit\_price is not None and exit\_reason is not None:
             self.logger.info(f"{exit\_reason} for position {position\_id} ({pair}) at price {exit\_price} on bar {bar\_timestamp}", source\_module=self.\_\_class\_\_.\_\_name\_\_)
             \# TODO: Add slippage to SL market order simulation?
             commission \= abs(entry\_qty \* exit\_price \* self.taker\_fee\_pct) \# Assume taker fee
             \_, quote\_asset \= pair.split("/")
             commission\_asset \= quote\_asset.upper()

             \# Publish execution report for the exit
             await self.\_publish\_simulated\_report(
                  originating\_event=originating\_event, \# Link back to original signal
                  status="FILLED",
                  qty\_filled=entry\_qty, \# Assume full exit for now
                  avg\_price=exit\_price,
                  commission=commission,
                  commission\_asset=commission\_asset,
                  error\_msg=exit\_reason, \# Use error\_msg to indicate SL/TP
                  fill\_timestamp=bar\_timestamp \# Timestamp of the triggering bar
             )
             triggered\_positions.append(position\_id) \# Mark for removal

    \# Remove triggered positions from active monitoring
    for pos\_id in triggered\_positions:
         if pos\_id in self.\_active\_sl\_tp:
              del self.\_active\_sl\_tp\[pos\_id\]
              self.logger.debug(f"Removed SL/TP monitoring for position {pos\_id}", source\_module=self.\_\_class\_\_.\_\_name\_\_)

### **B. Add Partial Fills Support (High Priority \- FR-607)**

* **Problem:** Simulation assumes orders fill completely if the price condition is met.
* **Solution:**
  1. **Modify Fill Logic:** In \_simulate\_market\_order and \_simulate\_limit\_order, estimate the fillable quantity based on the available liquidity in the next\_bar.
  2. **Liquidity Model:** Use a simple model initially: assume only a fraction of the bar's volume is available at the fill price level (configurable percentage, e.g., backtest.fill\_liquidity\_ratio).
  3. **Calculate Fill:** fill\_qty \= min(event.quantity, bar\['volume'\] \* fill\_liquidity\_ratio).
  4. **Update Status:** If fill\_qty \< event.quantity and fill\_qty \> 0, set status to "PARTIALLY\_FILLED". If fill\_qty \== event.quantity, set status to "FILLED". If fill\_qty \== 0 (e.g., limit not met or zero liquidity), set status to "REJECTED" or "NEW" (if it should remain open).
  5. **Publish Report:** Publish the ExecutionReportEvent with the actual qty\_filled.
  6. **Remaining Quantity:** The backtesting engine or portfolio manager needs to handle the remaining quantity of partially filled orders (e.g., keep trying to fill on subsequent bars, cancel, etc.). This simulation handler currently doesn't manage pending orders across multiple bars.

\# In SimulatedExecutionHandler class

\# Add config in \_\_init\_\_:
\# self.\_fill\_liquidity\_ratio \= self.config.get\_decimal("backtest.fill\_liquidity\_ratio", Decimal("0.1")) \# Fill up to 10% of bar volume

async def \_simulate\_market\_order(self, event: "TradeSignalApprovedEvent", next\_bar: pd.Series, ...) \-\> dict:
    \# ... (calculate base price, slippage, simulated\_fill\_price) ...

    \# \--- Partial Fill Logic \---
    available\_volume \= Decimal(str(next\_bar\["volume"\]))
    max\_fillable\_qty \= available\_volume \* self.\_fill\_liquidity\_ratio
    fill\_qty \= min(event.quantity, max\_fillable\_qty)
    status \= "REJECTED" \# Default if qty is 0
    if fill\_qty \> Decimal("1e-12"): \# Use threshold comparison
         if fill\_qty \< event.quantity:
              status \= "PARTIALLY\_FILLED"
              self.logger.info(f"Market order {event.signal\_id} partially filled: {fill\_qty}/{event.quantity} based on bar volume {available\_volume}", source\_module=self.\_\_class\_\_.\_\_name\_\_)
         else:
              status \= "FILLED"
    else:
         fill\_qty \= Decimal(0) \# Ensure it's zero if rejected
         self.logger.warning(f"Market order {event.signal\_id} rejected due to zero fillable quantity (Bar Volume: {available\_volume})", source\_module=self.\_\_class\_\_.\_\_name\_\_)

    \# \--- End Partial Fill Logic \---

    \# Calculate commission based on actual fill\_qty
    fill\_value \= fill\_qty \* simulated\_fill\_price
    commission\_amount \= abs(fill\_value \* self.taker\_fee\_pct)
    \# ... (get commission\_asset) ...

    return {
        "status": status,
        "quantity": fill\_qty, \# Return actual filled quantity
        "fill\_price": simulated\_fill\_price if fill\_qty \> 0 else None,
        "commission": commission\_amount,
        "commission\_asset": commission\_asset if fill\_qty \> 0 else None,
        "timestamp": fill\_timestamp,
        "error\_msg": None if status \!= "REJECTED" else "Zero fillable quantity"
    }

async def \_simulate\_limit\_order(self, event: "TradeSignalApprovedEvent", next\_bar: pd.Series, ...) \-\> Optional\[dict\]:
    \# ... (check limit price, handle missing price) ...
    filled\_price\_level \= self.\_check\_limit\_order\_fill(event.side, limit\_price, next\_bar)

    if not filled\_price\_level:
         \# ... (publish REJECTED report \- no fill) ...
         return None \# Indicate no fill occurred

    \# \--- Partial Fill Logic \---
    available\_volume \= Decimal(str(next\_bar\["volume"\]))
    \# Assume limit orders have lower priority or access less volume? Configurable.
    max\_fillable\_qty \= available\_volume \* self.\_fill\_liquidity\_ratio \* Decimal("0.5") \# Example: Limit gets 50% of market's share
    fill\_qty \= min(event.quantity, max\_fillable\_qty)
    status \= "REJECTED"
    if fill\_qty \> Decimal("1e-12"):
         if fill\_qty \< event.quantity:
              status \= "PARTIALLY\_FILLED"
              self.logger.info(f"Limit order {event.signal\_id} partially filled: {fill\_qty}/{event.quantity} based on bar volume {available\_volume}", source\_module=self.\_\_class\_\_.\_\_name\_\_)
         else:
              status \= "FILLED"
    else:
         fill\_qty \= Decimal(0)
         self.logger.warning(f"Limit order {event.signal\_id} rejected due to zero fillable quantity (Price Met, Bar Volume: {available\_volume})", source\_module=self.\_\_class\_\_.\_\_name\_\_)

    \# \--- End Partial Fill Logic \---

    \# Calculate commission based on actual fill\_qty
    fill\_value \= fill\_qty \* limit\_price \# Limit orders fill at limit price
    commission\_amount \= abs(fill\_value \* self.taker\_fee\_pct) \# Assume taker for simplicity, could be maker
    \# ... (get commission\_asset) ...

    return {
        "status": status,
        "quantity": fill\_qty, \# Return actual filled quantity
        "fill\_price": limit\_price if fill\_qty \> 0 else None,
        "commission": commission\_amount,
        "commission\_asset": commission\_asset if fill\_qty \> 0 else None,
        "timestamp": fill\_timestamp,
        "error\_msg": None if status \!= "REJECTED" else "Zero fillable quantity at limit price"
    }

### **C. Implement Limit Order Timeout (High Priority \- FR-605)**

* **Problem:** Limit orders are only checked against the *single* next bar.
* **Solution:**
  1. Modify \_simulate\_limit\_order (or the calling logic in handle\_trade\_signal\_approved).
  2. Fetch *multiple* subsequent bars using data\_service.get\_future\_bars(..., limit=timeout\_bars) where timeout\_bars is configurable (backtest.limit\_order\_timeout\_bars).
  3. Iterate through these bars. If \_check\_limit\_order\_fill returns true for any bar within the timeout window, simulate the fill using that bar's data and timestamp.
  4. If the loop completes without a fill, publish a "REJECTED" or "CANCELED" (due to timeout) execution report.

*(Note: This requires the HistoricalDataService to have a method like get\_future\_bars or the backtesting engine to manage the bar iteration and call the handler repeatedly.)*

### **D. Improve Slippage Model (Medium Priority)**

* **Problem:** Current slippage model (fixed %, ATR) is basic.
* **Solution:** Add a new slippage model option (e.g., market\_impact) in \_calculate\_slippage.
  * This model could estimate slippage based on the event.quantity relative to the next\_bar\['volume'\] or average volume fetched from data\_service.
  * Use a non-linear formula: slippage\_pct \= base\_slippage\_pct \+ impact\_factor \* (order\_qty / bar\_volume) \*\* exponent. Parameters (impact\_factor, exponent) should be configurable.

### **E. Standardize Error Handling & Add Validation (Medium Priority)**

* **Problem:** Inconsistent error handling; limited validation of incoming signals.
* **Solution:**
  1. **Standardize:** Use a consistent approach. Prefer logging errors and publishing "ERROR" or "REJECTED" ExecutionReportEvents rather than raising exceptions that might halt the simulation. Create a helper \_handle\_simulation\_error method.
  2. **Validation:** Add a \_validate\_order\_parameters method called at the start of handle\_trade\_signal\_approved. Check event.quantity against configured min/max order sizes for the pair (requires adding this info to config or fetching from data\_service/ExecutionHandler info). Check for valid side, order\_type, presence of limit\_price for limit orders, etc. Publish a "REJECTED" report if validation fails.

**Conclusion:** Implementing SL/TP simulation, partial fills, and limit order timeouts are the most critical steps to make the backtesting simulation significantly more realistic and compliant with requirements. Improving the slippage model and standardizing error handling/validation will further enhance its quality.
