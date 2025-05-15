# **Execution Handler: Review Summary & Solution Whiteboard**

## **1\. Summary of Manual Review Findings (execution\_handler.md)**

The review of src/gal\_friday/execution\_handler.py identified the following:

* **Strengths:** Comprehensive error handling for API calls, strong type hinting, robust Kraken authentication, graceful service lifecycle (start/stop), good validation of incoming signals, respects HALT condition.
* **Functionality Gaps (High Priority):**
  * **SL/TP Handling:** Acknowledged in signals but not implemented (FR-606).
  * **Order Timeout Logic:** Missing logic to cancel stale limit orders (FR-605).
  * **Order Status Monitoring:** No continuous monitoring after initial placement (FR-608).
  * **Partial Fill Handling:** Not explicitly handled (FR-607).
  * **WebSocket Implementation:** Placeholders exist, but no real-time updates via WebSocket (FR-603).
* **Error Handling & Robustness (Medium Priority):**
  * **Limited Retry Logic:** No exponential backoff for transient API errors.
  * **No Circuit Breaker:** Risk of flooding API during persistent issues.
  * **Rudimentary Rate Limit Handling:** No explicit tracking/throttling.
* **Code Organization:** Kraken-specific logic not abstracted, debug print statement present, some long methods.
* **Security:** Minor concerns about API key validation, potential logging of sensitive data, default SSL verification.

## **2\. Whiteboard: Proposed Solutions**

Here's a breakdown of solutions addressing the high and medium priority recommendations:

### **A. Implement Order Status Monitoring & Partial Fill Handling (High Priority \- FR-607, FR-608)**

* **Problem:** Orders are placed, but their subsequent status (fills, partial fills, cancellation) isn't tracked or reported after the initial "NEW" status.
* **Solution:**
  1. **Polling Task:** After successfully placing an order (\_handle\_add\_order\_response), launch a dedicated asyncio.Task to periodically poll the QueryOrders private endpoint for that specific order ID (txid).
  2. **Polling Logic (\_monitor\_order\_status):**
     * Use the txid obtained from the AddOrder response.
     * Call \_make\_private\_request\_with\_retry (see Section B) to query /0/private/QueryOrders with txid=\<kraken\_order\_id\>\&trades=true.
     * Parse the response, extracting status (open, closed, canceled, expired), filled quantity (vol\_exec), average fill price (price), fee (fee), and individual trades if needed for partial fills.
     * Compare the current state with the previously known state (may need to store order state locally).
     * If a change is detected (e.g., status change, increase in vol\_exec), publish a new ExecutionReportEvent with the updated details (including quantity\_filled, average\_fill\_price, commission).
     * Handle partial fills by publishing reports for each increment of quantity\_filled.
     * Stop polling when the order reaches a terminal state (closed, canceled, expired).
     * Use appropriate delays between polls (configurable).
     * Handle errors during polling (e.g., order not found after initial placement).
  3. **Trigger SL/TP:** Once an order is fully filled (status \== 'closed' and vol\_exec matches quantity\_ordered), trigger the SL/TP placement logic (see Section C).

\# In ExecutionHandler class

\# Store active monitoring tasks
\# self.\_order\_monitoring\_tasks: Dict\[str, asyncio.Task\] \= {} \# txid \-\> Task

async def \_handle\_add\_order\_response(self, result: Dict\[str, Any\], originating\_event: TradeSignalApprovedEvent, cl\_ord\_id: str) \-\> None:
    \# ... (existing success logic before publishing NEW report) ...
    if txids and isinstance(txids, list):
        kraken\_order\_id \= txids\[0\]
        \# ... (log success, store order map) ...

        \# Publish initial "NEW" report (as before)
        \# ...

        \# \--- START MONITORING \---
        monitor\_task \= asyncio.create\_task(
            self.\_monitor\_order\_status(kraken\_order\_id, cl\_ord\_id, originating\_event.signal\_id)
        )
        \# Store task reference if needed for cancellation on stop
        \# self.\_order\_monitoring\_tasks\[kraken\_order\_id\] \= monitor\_task
        \# \--- END MONITORING \---
    \# ... (existing error handling) ...

async def \_monitor\_order\_status(self, exchange\_order\_id: str, client\_order\_id: str, signal\_id: Optional\[UUID\]) \-\> None:
    """Monitors the status of a specific order via polling."""
    self.logger.info(f"Starting status monitoring for order {exchange\_order\_id} (cl={client\_order\_id})", source\_module=self.\_source\_module)
    \# Configurable parameters
    poll\_interval \= self.config.get\_float("order.status\_poll\_interval\_s", 5.0)
    max\_poll\_duration \= self.config.get\_float("order.max\_poll\_duration\_s", 3600.0) \# e.g., 1 hour
    start\_time \= time.time()
    last\_known\_status: Optional\[str\] \= "NEW"
    last\_known\_filled\_qty: Decimal \= Decimal(0)

    while time.time() \- start\_time \< max\_poll\_duration:
        await asyncio.sleep(poll\_interval)

        uri\_path \= "/0/private/QueryOrders"
        params \= {"txid": exchange\_order\_id, "trades": "true"} \# Include trade info

        \# Use retry logic for the query itself
        query\_result \= await self.\_make\_private\_request\_with\_retry(uri\_path, params) \# See Section B

        if not query\_result or query\_result.get("error"):
            error\_str \= str(query\_result.get("error", "Unknown query error"))
            self.logger.error(f"Error querying order {exchange\_order\_id}: {error\_str}", source\_module=self.\_source\_module)
            \# Decide if error is fatal for monitoring this order
            if "EOrder:Unknown order" in error\_str: \# Example fatal error
                 self.logger.error(f"Order {exchange\_order\_id} not found. Stopping monitoring.", source\_module=self.\_source\_module)
                 break
            continue \# Continue polling on potentially transient errors

        order\_data \= query\_result.get("result", {}).get(exchange\_order\_id)
        if not order\_data:
            self.logger.warning(f"Order {exchange\_order\_id} not found in QueryOrders result. Retrying.", source\_module=self.\_source\_module)
            continue

        \# \--- Process Order Data \---
        current\_status \= order\_data.get("status")
        current\_filled\_qty\_str \= order\_data.get("vol\_exec", "0")
        avg\_fill\_price\_str \= order\_data.get("price") \# Average price for filled portion
        fee\_str \= order\_data.get("fee")
        \# TODO: Parse commission asset from trade details if needed

        try:
            current\_filled\_qty \= Decimal(current\_filled\_qty\_str)
            avg\_fill\_price \= Decimal(avg\_fill\_price\_str) if avg\_fill\_price\_str else None
            commission \= Decimal(fee\_str) if fee\_str else None
        except Exception as e:
             self.logger.error(f"Error parsing numeric data for order {exchange\_order\_id}: {e}. Data: {order\_data}", source\_module=self.\_source\_module)
             continue \# Skip update if parsing fails

        \# \--- Check for Changes and Publish \---
        status\_changed \= current\_status \!= last\_known\_status
        fill\_increased \= current\_filled\_qty \> last\_known\_filled\_qty

        if status\_changed or fill\_increased:
            self.logger.info(f"Status change for {exchange\_order\_id}: Status='{current\_status}', Filled={current\_filled\_qty}. Publishing report.", source\_module=self.\_source\_module)
            \# Publish ExecutionReportEvent with updated data
            await self.\_publish\_status\_execution\_report(
                exchange\_order\_id=exchange\_order\_id,
                client\_order\_id=client\_order\_id,
                signal\_id=signal\_id,
                order\_data=order\_data, \# Pass raw data for easier field access
                current\_status=current\_status,
                current\_filled\_qty=current\_filled\_qty,
                avg\_fill\_price=avg\_fill\_price,
                commission=commission
            )
            last\_known\_status \= current\_status
            last\_known\_filled\_qty \= current\_filled\_qty

            \# \--- Trigger SL/TP on Full Fill \---
            \# Check if fully filled (status 'closed' and volume matches original \- need original event/qty)
            \# if current\_status \== 'closed' and current\_filled\_qty \>= original\_quantity:
            \#    self.logger.info(f"Order {exchange\_order\_id} fully filled. Triggering SL/TP handling.", source\_module=self.\_source\_module)
            \#    asyncio.create\_task(self.\_handle\_sl\_tp\_orders(originating\_event, exchange\_order\_id)) \# Need originating\_event

        \# \--- Stop Monitoring Condition \---
        if current\_status in \["closed", "canceled", "expired"\]:
            self.logger.info(f"Order {exchange\_order\_id} reached terminal state '{current\_status}'. Stopping monitoring.", source\_module=self.\_source\_module)
            break \# Exit loop

    else: \# Loop finished due to timeout
         self.logger.warning(f"Stopped monitoring order {exchange\_order\_id} after timeout ({max\_poll\_duration}s). Last status: {last\_known\_status}", source\_module=self.\_source\_module)

    \# Clean up task reference if stored
    \# self.\_order\_monitoring\_tasks.pop(exchange\_order\_id, None)

async def \_publish\_status\_execution\_report(self, exchange\_order\_id: str, client\_order\_id: str, signal\_id: Optional\[UUID\], order\_data: Dict, current\_status: str, current\_filled\_qty: Decimal, avg\_fill\_price: Optional\[Decimal\], commission: Optional\[Decimal\]) \-\> None:
    """Helper to publish ExecutionReportEvent based on polled status."""
    try:
        \# Extract necessary fields from order\_data (Kraken specific)
        descr \= order\_data.get("descr", {})
        order\_type \= descr.get("ordertype")
        side \= descr.get("type")
        pair \= descr.get("pair") \# Kraken pair name
        \# Need to map pair back to internal name if possible/necessary
        internal\_pair \= self.\_map\_kraken\_pair\_to\_internal(pair) if pair else "UNKNOWN"
        quantity\_ordered\_str \= order\_data.get("vol")
        limit\_price\_str \= descr.get("price") \# Price for limit orders

        quantity\_ordered \= Decimal(quantity\_ordered\_str) if quantity\_ordered\_str else Decimal(0)
        limit\_price \= Decimal(limit\_price\_str) if limit\_price\_str else None
        \# Determine commission asset (often the quote currency)
        commission\_asset \= self.\_get\_quote\_currency(internal\_pair) if internal\_pair \!= "UNKNOWN" else None

        report \= ExecutionReportEvent(
            source\_module=self.\_\_class\_\_.\_\_name\_\_,
            event\_id=uuid.uuid4(), \# Generate new UUID for each report
            timestamp=datetime.utcnow(),
            signal\_id=signal\_id,
            exchange\_order\_id=exchange\_order\_id,
            client\_order\_id=client\_order\_id,
            trading\_pair=internal\_pair,
            exchange=self.config.get("exchange.name", "kraken"),
            order\_status=current\_status.upper(), \# Standardize status
            order\_type=order\_type.upper() if order\_type else "UNKNOWN", \# Standardize type
            side=side.upper() if side else "UNKNOWN", \# Standardize side
            quantity\_ordered=quantity\_ordered,
            quantity\_filled=current\_filled\_qty,
            limit\_price=limit\_price,
            average\_fill\_price=avg\_fill\_price,
            commission=commission,
            commission\_asset=commission\_asset,
            timestamp\_exchange=datetime.fromtimestamp(order\_data.get("opentm", time.time()), tz=datetime.timezone.utc) if order\_data.get("opentm") else None, \# Example: use open time
            error\_message=order\_data.get("reason"), \# Include reason if status is 'canceled' or 'expired'
        )
        await self.pubsub.publish(report)
    except Exception as e:
        self.logger.error(f"Failed to publish status execution report for {exchange\_order\_id}: {e}", source\_module=self.\_source\_module, exc\_info=True)

\# \--- Helper methods needed \---
def \_map\_kraken\_pair\_to\_internal(self, kraken\_pair: str) \-\> Optional\[str\]:
     """Maps Kraken pair name (e.g., XXBTZUSD) back to internal name (e.g., BTC/USD)."""
     for internal\_name, info in self.\_pair\_info.items():
         if info.get('altname') \== kraken\_pair or info.get('wsname') \== kraken\_pair or info.get('kraken\_pair\_key') \== kraken\_pair:
             return internal\_name
     self.logger.warning(f"Could not map Kraken pair '{kraken\_pair}' back to internal name.", source\_module=self.\_source\_module)
     return None

def \_get\_quote\_currency(self, internal\_pair: str) \-\> Optional\[str\]:
     """Gets the quote currency for an internal pair name."""
     info \= self.\_pair\_info.get(internal\_pair)
     return info.get('quote') if info else None

### **B. Implement Retry Logic (High Priority \- FR-609)**

* **Problem:** Transient network issues or temporary API errors aren't retried, leading to failed actions.
* **Solution:** Wrap the \_make\_private\_request call within a retry loop (\_make\_private\_request\_with\_retry). Implement exponential backoff with jitter. Define which errors are considered "retryable" (e.g., connection errors, timeouts, specific Kraken temporary error codes like EService:Unavailable) vs. permanent (e.g., EOrder:Invalid arguments, EGeneral:Permission denied).
  \# In ExecutionHandler class
  import random

  def \_is\_retryable\_error(self, error\_str: str) \-\> bool:
      """Check if a Kraken error string indicates a potentially transient issue."""
      \# Add known transient error codes/messages from Kraken docs
      retryable\_codes \= \[
          "EGeneral:Temporary", "EService:Unavailable", "EService:Busy",
          "EGeneral:Timeout", "EGeneral:ConnectionError", \# Internal codes added
          \# Add specific Kraken codes if known e.g., EAPI:Rate limit exceeded (though rate limit needs specific handling)
      \]
      \# Simple check, could be more sophisticated
      return any(code in error\_str for code in retryable\_codes)

  async def \_make\_private\_request\_with\_retry(self, uri\_path: str, data: Dict\[str, Any\], max\_retries: int \= 3\) \-\> Dict\[str, Any\]:
      """Makes a private request with retry logic for transient errors."""
      base\_delay \= self.config.get\_float("exchange.retry\_base\_delay\_s", 1.0)
      last\_exception \= None

      for attempt in range(max\_retries \+ 1):
          try:
              result \= await self.\_make\_private\_request(uri\_path, data)
              last\_exception \= None \# Clear last exception on successful call structure

              \# Check for API-level errors within the result
              if result.get("error"):
                  error\_str \= str(result\["error"\])
                  if self.\_is\_retryable\_error(error\_str) and attempt \< max\_retries:
                      delay \= min(base\_delay \* (2 \*\* attempt), 30.0) \# Cap delay
                      jitter \= random.uniform(0, delay \* 0.1)
                      total\_delay \= delay \+ jitter
                      self.logger.warning(f"Retryable API error for {uri\_path}: {error\_str}. Retrying in {total\_delay:.2f}s (Attempt {attempt \+ 1}/{max\_retries \+ 1})", source\_module=self.\_source\_module)
                      await asyncio.sleep(total\_delay)
                      continue \# Go to next attempt
                  else:
                      \# Permanent error or max retries reached
                      return result
              else:
                  \# Successful API call (no 'error' field or empty error list)
                  return result

          except Exception as e:
               \# Catch exceptions from \_make\_private\_request itself (like connection errors handled there)
               \# This outer catch is more for unexpected issues during the retry loop logic
               self.logger.error(f"Unexpected exception during retry loop for {uri\_path}: {e}", source\_module=self.\_source\_module, exc\_info=True)
               last\_exception \= e
               \# Decide if the exception type itself is retryable (e.g. network issues)
               \# For simplicity now, we rely on \_make\_private\_request standardizing errors
               \# Break or continue based on exception type if needed. If we break:
               \# break

      \# If loop finishes, all retries failed
      self.logger.error(f"API request to {uri\_path} failed after {max\_retries \+ 1} attempts.", source\_module=self.\_source\_module)
      \# Return the last known error result or a generic max retries error
      if isinstance(last\_exception, Exception):
           \# If the loop exited due to an uncaught exception in the retry logic itself
            return {"error": \[f"EGeneral:UnexpectedRetryFailure \- {last\_exception}"\]}
      \# If the loop finished because the last attempt resulted in a permanent error or max retries on a retryable error
      \# The 'result' from the last iteration should contain the final error. If result wasn't assigned (e.g. first attempt exception), create generic error.
      \# This part needs careful handling based on exact flow. Assume last 'result' holds the final error.
      \# If 'result' is not defined here, create a generic error:
      \# return {"error": \["EGeneral:MaxRetriesExceeded"\]}
      \# Assuming 'result' holds the last error:
      return result if 'result' in locals() else {"error": \["EGeneral:MaxRetriesExceeded"\]}

  \# \--- Update calling methods \---
  \# Replace calls to \_make\_private\_request with \_make\_private\_request\_with\_retry
  \# Example in handle\_trade\_signal\_approved:
  \# result \= await self.\_make\_private\_request\_with\_retry(uri\_path, kraken\_params)
  \# Example in \_monitor\_order\_status:
  \# query\_result \= await self.\_make\_private\_request\_with\_retry(uri\_path, params)

### **C. Implement SL/TP Order Management (High Priority \- FR-606)**

* **Problem:** SL/TP prices are included in the approved signal but aren't used to place corresponding orders.
* **Solution:**
  1. **Trigger:** After the primary entry order is confirmed *fully filled* (via the status monitoring task), trigger a new function \_handle\_sl\_tp\_orders.
  2. **Parameter Preparation:** Create separate parameter dictionaries for the SL and TP orders based on the original TradeSignalApprovedEvent and the fill details (quantity, side).
     * SL order type: stop-loss-limit or stop-loss (Kraken specific). Requires price (stop price \= sl\_price) and potentially price2 (limit price).
     * TP order type: take-profit-limit or take-profit. Requires price (take-profit price \= tp\_price) and potentially price2.
     * Volume should match the filled quantity of the parent order.
     * Side will be opposite to the parent order.
     * Consider using reduce-only flag if appropriate.
  3. **Placement:** Use \_make\_private\_request\_with\_retry to call AddOrder for the SL and TP orders separately.
  4. **Linking/Reporting:** Publish ExecutionReportEvents for these new SL/TP orders. Potentially link them back to the original signal\_id. Monitor their status as well. *Complexity Note:* Managing One-Cancels-the-Other (OCO) logic for SL/TP via REST is complex; Kraken might offer specific order types or WebSocket features for this. The simplest MVP is placing independent SL and TP orders.

\# In ExecutionHandler class

async def \_handle\_sl\_tp\_orders(self, originating\_event: TradeSignalApprovedEvent, filled\_order\_id: str, filled\_quantity: Decimal) \-\> None:
    """Places SL and/or TP orders contingent on the filled entry order."""
    self.logger.info(f"Handling SL/TP placement for filled order {filled\_order\_id} (Signal: {originating\_event.signal\_id})", source\_module=self.\_source\_module)

    kraken\_pair\_name \= self.\_get\_kraken\_pair\_name(originating\_event.trading\_pair)
    if not kraken\_pair\_name: return \# Error logged in helper

    \# Determine side for SL/TP (opposite of entry)
    exit\_side \= "sell" if originating\_event.side \== "BUY" else "buy"

    \# \--- Place Stop Loss Order \---
    if originating\_event.sl\_price:
        sl\_params \= self.\_prepare\_contingent\_order\_params(
            pair=kraken\_pair\_name,
            side=exit\_side,
            order\_type="stop-loss", \# Or stop-loss-limit
            price=originating\_event.sl\_price, \# Stop price
            \# price2=... \# Optional limit price for stop-loss-limit
            volume=filled\_quantity,
            pair\_info=self.\_pair\_info.get(originating\_event.trading\_pair),
            signal\_id=originating\_event.signal\_id,
            contingent\_type="SL"
        )
        if sl\_params:
            sl\_cl\_ord\_id \= f"gf-sl-{str(originating\_event.signal\_id)\[:8\]}-{int(time.time() \* 1000000)}"
            sl\_params\["cl\_ord\_id"\] \= sl\_cl\_ord\_id
            sl\_params\["reduce\_only"\] \= "true" \# Good practice for exits
            self.logger.info(f"Placing SL order for signal {originating\_event.signal\_id} with cl\_ord\_id {sl\_cl\_ord\_id}", source\_module=self.\_source\_module)
            sl\_result \= await self.\_make\_private\_request\_with\_retry("/0/private/AddOrder", sl\_params)
            \# Handle SL order placement response (publish report, start monitoring)
            await self.\_handle\_add\_order\_response(sl\_result, originating\_event, sl\_cl\_ord\_id) \# May need adjustment for SL/TP context

    \# \--- Place Take Profit Order \---
    if originating\_event.tp\_price:
        tp\_params \= self.\_prepare\_contingent\_order\_params(
            pair=kraken\_pair\_name,
            side=exit\_side,
            order\_type="take-profit-limit", \# Or take-profit
            price=originating\_event.tp\_price, \# Limit price for TP
            \# price2=... \# Optional trigger price for take-profit
            volume=filled\_quantity,
            pair\_info=self.\_pair\_info.get(originating\_event.trading\_pair),
            signal\_id=originating\_event.signal\_id,
            contingent\_type="TP"
        )
        if tp\_params:
            tp\_cl\_ord\_id \= f"gf-tp-{str(originating\_event.signal\_id)\[:8\]}-{int(time.time() \* 1000000)}"
            tp\_params\["cl\_ord\_id"\] \= tp\_cl\_ord\_id
            tp\_params\["reduce\_only"\] \= "true" \# Good practice for exits
            self.logger.info(f"Placing TP order for signal {originating\_event.signal\_id} with cl\_ord\_id {tp\_cl\_ord\_id}", source\_module=self.\_source\_module)
            tp\_result \= await self.\_make\_private\_request\_with\_retry("/0/private/AddOrder", tp\_params)
            \# Handle TP order placement response (publish report, start monitoring)
            await self.\_handle\_add\_order\_response(tp\_result, originating\_event, tp\_cl\_ord\_id) \# May need adjustment for SL/TP context

def \_prepare\_contingent\_order\_params(self, pair: str, side: str, order\_type: str, price: Decimal, volume: Decimal, pair\_info: Optional\[Dict\], signal\_id: UUID, contingent\_type: str, price2: Optional\[Decimal\] \= None) \-\> Optional\[Dict\[str, Any\]\]:
    """Helper to prepare parameters for SL/TP orders, including validation."""
    params \= {"pair": pair, "type": side, "ordertype": order\_type}

    if not pair\_info:
         self.logger.error(f"Missing pair\_info for contingent order {contingent\_type} (Signal: {signal\_id})", source\_module=self.\_source\_module)
         return None

    \# Validate and format volume
    lot\_decimals \= pair\_info.get("lot\_decimals")
    if lot\_decimals is None: \# Basic check
         self.logger.error(f"Missing lot\_decimals for contingent order {contingent\_type} (Signal: {signal\_id})", source\_module=self.\_source\_module)
         return None
    try:
        params\["volume"\] \= self.\_format\_decimal(volume, lot\_decimals)
    except Exception as e:
         self.logger.error(f"Error formatting volume for contingent order {contingent\_type} (Signal: {signal\_id}): {e}", source\_module=self.\_source\_module)
         return None

    \# Validate and format price(s)
    pair\_decimals \= pair\_info.get("pair\_decimals")
    if pair\_decimals is None:
         self.logger.error(f"Missing pair\_decimals for contingent order {contingent\_type} (Signal: {signal\_id})", source\_module=self.\_source\_module)
         return None
    try:
        params\["price"\] \= self.\_format\_decimal(price, pair\_decimals)
        if price2 is not None:
            params\["price2"\] \= self.\_format\_decimal(price2, pair\_decimals)
    except Exception as e:
         self.logger.error(f"Error formatting price for contingent order {contingent\_type} (Signal: {signal\_id}): {e}", source\_module=self.\_source\_module)
         return None

    \# Add other necessary parameters (e.g., timeinforce if needed)
    return params

def \_get\_kraken\_pair\_name(self, internal\_pair: str) \-\> Optional\[str\]:
    """Helper to get the Kraken pair name from stored info."""
    info \= self.\_pair\_info.get(internal\_pair)
    name \= info.get('altname') if info else None
    if not name:
         self.logger.error(f"Could not find Kraken pair name for internal pair '{internal\_pair}'", source\_module=self.\_source\_module)
    return name

\# Modify \_monitor\_order\_status to trigger \_handle\_sl\_tp\_orders
async def \_monitor\_order\_status(self, exchange\_order\_id: str, client\_order\_id: str, signal\_id: Optional\[UUID\]) \-\> None:
    \# ... inside the loop, after processing order data ...
    \# Need access to the original event or at least the original quantity
    \# This requires passing more context into the monitor task or fetching it.
    \# Assuming 'originating\_event' is available here for simplicity:
    originating\_event: Optional\[TradeSignalApprovedEvent\] \= await self.\_get\_originating\_signal\_event(signal\_id) \# Hypothetical fetch

    if originating\_event and current\_status \== 'closed' and current\_filled\_qty \>= originating\_event.quantity:
        if not await self.\_has\_sl\_tp\_been\_placed(signal\_id): \# Need mechanism to track this
            self.logger.info(f"Order {exchange\_order\_id} fully filled. Triggering SL/TP handling.", source\_module=self.\_source\_module)
            asyncio.create\_task(self.\_handle\_sl\_tp\_orders(originating\_event, exchange\_order\_id, current\_filled\_qty))
            await self.\_mark\_sl\_tp\_as\_placed(signal\_id) \# Mark as placed

    \# ... rest of the loop ...

\# \--- Need helper state/methods for SL/TP tracking \---
\# self.\_placed\_sl\_tp\_signals: Set\[UUID\] \= set()
\# async def \_has\_sl\_tp\_been\_placed(self, signal\_id: Optional\[UUID\]) \-\> bool: ...
\# async def \_mark\_sl\_tp\_as\_placed(self, signal\_id: Optional\[UUID\]) \-\> None: ...
\# async def \_get\_originating\_signal\_event(self, signal\_id: Optional\[UUID\]) \-\> Optional\[TradeSignalApprovedEvent\]: ... \# Needs event store or cache

### **D. Implement Rate Limit Tracking (Medium Priority)**

* **Problem:** Risk of exceeding API rate limits, leading to temporary blocks.
* **Solution:** Implement a RateLimitTracker class (similar to the one suggested in the review). Before calling \_make\_private\_request (or \_with\_retry), call await self.rate\_limiter.wait\_for\_private\_capacity(). This requires initializing the tracker in \_\_init\_\_ and potentially fetching limits based on API key tier if possible.
  \# In \_\_init\_\_
  \# self.rate\_limiter \= RateLimitTracker(self.config) \# Assuming RateLimitTracker class exists

  \# In \_make\_private\_request\_with\_retry (or before calling it)
  \# await self.rate\_limiter.wait\_for\_private\_capacity() \# Add this before making the actual request
  \# result \= await self.\_make\_private\_request(uri\_path, data)

### **E. Implement Limit Order Timeout (Medium Priority \- FR-605)**

* **Problem:** Limit orders might stay open indefinitely if not filled.
* **Solution:**
  1. When placing a limit order, determine the timeout duration from config.
  2. Launch a separate asyncio.Task (\_monitor\_limit\_order\_timeout) that sleeps for the timeout duration.
  3. After waking up, the task checks the order status using QueryOrders.
  4. If the order is still open (or pending), call CancelOrder using the txid.
  5. Publish an ExecutionReportEvent with status CANCELED and reason "Timeout".

 \# In ExecutionHandler class

 \# Modify \_handle\_add\_order\_response for limit orders
 async def \_handle\_add\_order\_response(self, result: Dict\[str, Any\], originating\_event: TradeSignalApprovedEvent, cl\_ord\_id: str) \-\> None:
     \# ... (after successful placement and getting kraken\_order\_id) ...
     if originating\_event.order\_type \== "LIMIT":
         timeout\_s \= self.config.get\_float("order.limit\_order\_timeout\_s", 300.0) \# e.g., 5 mins
         if timeout\_s \> 0:
              self.logger.info(f"Scheduling timeout check for limit order {kraken\_order\_id} in {timeout\_s}s.", source\_module=self.\_source\_module)
              asyncio.create\_task(
                  self.\_monitor\_limit\_order\_timeout(kraken\_order\_id, cl\_ord\_id, timeout\_s)
              )
     \# ...

 async def \_monitor\_limit\_order\_timeout(self, exchange\_order\_id: str, client\_order\_id: str, timeout\_seconds: float) \-\> None:
     """Checks if a limit order is filled after a timeout and cancels if not."""
     await asyncio.sleep(timeout\_seconds)
     self.logger.info(f"Timeout reached for limit order {exchange\_order\_id}. Checking status.", source\_module=self.\_source\_module)

     uri\_path \= "/0/private/QueryOrders"
     params \= {"txid": exchange\_order\_id}
     query\_result \= await self.\_make\_private\_request\_with\_retry(uri\_path, params)

     if not query\_result or query\_result.get("error"):
         self.logger.error(f"Error querying order {exchange\_order\_id} for timeout check: {query\_result.get('error', 'Unknown query error')}", source\_module=self.\_source\_module)
         return \# Cannot determine status, don't cancel arbitrarily

     order\_data \= query\_result.get("result", {}).get(exchange\_order\_id)
     if not order\_data:
         self.logger.warning(f"Order {exchange\_order\_id} not found during timeout check (already closed/canceled?).", source\_module=self.\_source\_module)
         return \# Order likely already closed or canceled

     status \= order\_data.get("status")
     if status in \["open", "pending"\]:
         self.logger.warning(f"Limit order {exchange\_order\_id} still '{status}' after {timeout\_seconds}s timeout. Attempting cancellation.", source\_module=self.\_source\_module)
         \# Call cancel\_order method (needs implementation)
         cancel\_success \= await self.cancel\_order(exchange\_order\_id)
         if not cancel\_success:
              self.logger.error(f"Failed to cancel timed-out limit order {exchange\_order\_id}.", source\_module=self.\_source\_module)
         \# The cancel\_order method should publish the CANCELED report
     else:
         self.logger.info(f"Limit order {exchange\_order\_id} already in terminal state '{status}' during timeout check.", source\_module=self.\_source\_module)

 async def cancel\_order(self, exchange\_order\_id: str) \-\> bool:
     """Cancels an open order on the exchange."""
     self.logger.info(f"Attempting to cancel order {exchange\_order\_id}", source\_module=self.\_source\_module)
     uri\_path \= "/0/private/CancelOrder"
     params \= {"txid": exchange\_order\_id}

     result \= await self.\_make\_private\_request\_with\_retry(uri\_path, params)

     if not result or result.get("error"):
         self.logger.error(f"Failed to cancel order {exchange\_order\_id}: {result.get('error', 'Unknown cancel error')}", source\_module=self.\_source\_module)
         return False

     \# Check response \- successful cancellation might have count \> 0
     count \= result.get("result", {}).get("count", 0\)
     if count \> 0:
         self.logger.info(f"Successfully initiated cancellation for order {exchange\_order\_id}. Count: {count}", source\_module=self.\_source\_module)
         \# Note: Cancellation might take time. The status monitor should pick up the 'canceled' status.
         \# Optionally, publish a CANCELED report immediately here, but relying on monitor might be better.
         \# await self.\_publish\_cancellation\_report(exchange\_order\_id, "Cancelled by system (timeout/request)")
         return True
     else:
          \# Order might have already been closed/canceled
          self.logger.warning(f"Cancellation request for {exchange\_order\_id} returned count 0\. Order might already be in terminal state.", source\_module=self.\_source\_module)
          \# Check 'pending' field in response if available
          return False \# Indicate cancellation wasn't actively performed now

 \# async def \_publish\_cancellation\_report(...) \# Helper if needed

### **F. Extract Kraken Logic & Refactor (Low/Medium Priority)**

* **Problem:** Code is specific to Kraken API. Debug print statement exists. Long methods.
* **Solution:**
  1. **Adapter Pattern:** Create a BaseExecutionAdapter abstract class defining methods like place\_order, cancel\_order, query\_order\_status, get\_exchange\_info. Implement a KrakenExecutionAdapter inheriting from the base class, moving all Kraken-specific URL paths, authentication, parameter translation, and response parsing into it. The ExecutionHandler then uses the adapter.
  2. **Logging:** Replace print("Execution Handler Loaded") with self.logger.info("Execution Handler module loaded", source\_module=\_\_name\_\_) or similar within \_\_init\_\_.
  3. **Refactor:** Break down methods like handle\_trade\_signal\_approved, \_translate\_signal\_to\_kraken\_params, and \_handle\_add\_order\_response into smaller, more focused helper functions.

Addressing the high-priority items (Monitoring, Retries, SL/TP) is crucial for meeting the functional requirements. Implementing rate limiting and the adapter pattern will significantly improve robustness and maintainability.
