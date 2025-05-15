# **Strategy Arbitrator (strategy\_arbitrator.py): Review Summary & Solution Whiteboard**

## **1\. Summary of Manual Review Findings (strategy\_arbitrator.md)**

* **Strengths:** Configurable strategy parameters (thresholds, entry type), clean event handling (sub/pub), basic type validation on incoming events, proper module lifecycle management (start/stop), includes strategy ID in proposals.
* **Functional Requirements Gaps (High Priority):**
  * **Missing SL/TP Calculation:** Uses placeholder Decimal("0") for proposed\_sl\_price and proposed\_tp\_price instead of calculating them based on config (sl\_pct, tp\_pct) and current price (FR-404).
  * **No Secondary Confirmation Logic:** Relies solely on prediction threshold, ignoring potential secondary checks like feature confirmation (FR-403).
  * **Missing Trade Exit Logic:** No logic implemented for triggering exits based on time, prediction reversal, etc. (FR-407, FR-408). *(Note: Exit logic might belong elsewhere, e.g., PortfolioManager or a dedicated ExitManager, but the arbitrator could potentially publish exit signals too).*
  * **No Entry Price Determination:** For limit orders, proposed\_entry\_price is left as None instead of being calculated (e.g., based on current bid/ask).
* **Design & Implementation Issues:** Debug print statement, unused PredictionPayload dataclass, error messages lack context, potentially outdated docstring comments.
* **Error Handling Concerns:** Limited validation of prediction event data, doesn't handle missing prediction values gracefully, raises exceptions directly during init instead of logging/handling.
* **Configuration & Hardcoding:** Assumes binary prediction P(down) \= 1 \- P(up), missing validation for all config parameters, assumes prediction\_value meaning (always P(up)?).

## **2\. Whiteboard: Proposed Solutions**

Addressing the high and medium priority recommendations:

### **A. Implement Proper SL/TP Price Calculation (High Priority \- FR-404)**

* **Problem:** proposed\_sl\_price and proposed\_tp\_price are placeholders.
* **Solution:**
  1. **Get Current Price:** The \_evaluate\_strategy method needs access to the current market price (e.g., mid-price, last trade price) to calculate SL/TP levels. This requires injecting or accessing the MarketPriceService.
  2. **Calculate:** Based on the signal side and the current\_price, calculate sl\_price \= current\_price \* (1 \+/- sl\_pct) and tp\_price \= current\_price \* (1 \+/- tp\_pct). Handle potential None values for configured percentages.
  3. **Update Event:** Populate the TradeSignalProposedEvent with the calculated Decimal values for proposed\_sl\_price and proposed\_tp\_price.

\# In StrategyArbitrator class

\# Add MarketPriceService dependency in \_\_init\_\_
\# def \_\_init\_\_(..., market\_price\_service: "MarketPriceService"):
\#    ...
\#    self.market\_price\_service \= market\_price\_service \# Requires injection
\#    ...

async def \_evaluate\_strategy( \# Make async if fetching price
    self, prediction\_event: PredictionEvent
) \-\> Optional\[TradeSignalProposedEvent\]:
    \# ... (determine side as before) ...

    if side:
        \# \--- Get Current Price \---
        current\_price \= await self.market\_price\_service.get\_latest\_price(prediction\_event.trading\_pair)
        if current\_price is None:
             self.logger.warning(f"Cannot generate signal for {prediction\_event.trading\_pair}: Failed to get current price.", source\_module=self.\_source\_module)
             return None
        \# \--- End Get Current Price \---

        \# \--- Calculate SL/TP \---
        sl\_price, tp\_price \= self.\_calculate\_sl\_tp\_prices(side, current\_price)
        if sl\_price is None or tp\_price is None:
             \# Error logged in helper
             return None
        \# \--- End Calculate SL/TP \---

        \# \--- Determine Entry Price (See Section C) \---
        proposed\_entry \= self.\_determine\_entry\_price(side, current\_price)
        \# \--- End Determine Entry Price \---

        signal\_id \= uuid.uuid4()
        proposed\_event \= TradeSignalProposedEvent(
            source\_module=self.\_source\_module,
            event\_id=uuid.uuid4(),
            timestamp=datetime.utcnow(),
            signal\_id=signal\_id,
            trading\_pair=prediction\_event.trading\_pair,
            exchange=prediction\_event.exchange,
            side=side,
            entry\_type=self.\_entry\_type,
            proposed\_sl\_price=sl\_price, \# Use calculated value
            proposed\_tp\_price=tp\_price, \# Use calculated value
            strategy\_id=self.\_strategy\_id,
            proposed\_entry\_price=proposed\_entry, \# Use calculated value
            triggering\_prediction\_event\_id=prediction\_event.event\_id,
        )
        \# ... (log and return event) ...
    \# ... (rest of method) ...

def \_calculate\_sl\_tp\_prices(self, side: str, current\_price: Decimal) \-\> Tuple\[Optional\[Decimal\], Optional\[Decimal\]\]:
    """Calculates SL/TP prices based on configuration and current price."""
    if self.\_sl\_pct is None or self.\_tp\_pct is None:
        self.logger.error("SL/TP percentages are not configured or loaded correctly.", source\_module=self.\_source\_module)
        return None, None
    if current\_price \<= 0:
         self.logger.error(f"Cannot calculate SL/TP: Invalid current\_price {current\_price}", source\_module=self.\_source\_module)
         return None, None

    try:
        if side \== "BUY":
            sl\_price \= current\_price \* (Decimal("1") \- self.\_sl\_pct)
            tp\_price \= current\_price \* (Decimal("1") \+ self.\_tp\_pct)
        elif side \== "SELL":
            sl\_price \= current\_price \* (Decimal("1") \+ self.\_sl\_pct)
            tp\_price \= current\_price \* (Decimal("1") \- self.\_tp\_pct)
        else:
            return None, None \# Should not happen

        \# Add rounding based on pair precision if needed
        \# sl\_price \= self.\_round\_price(sl\_price, trading\_pair)
        \# tp\_price \= self.\_round\_price(tp\_price, trading\_pair)

        return sl\_price, tp\_price
    except Exception as e:
        self.logger.error(f"Error calculating SL/TP prices: {e}", exc\_info=True, source\_module=self.\_source\_module)
        return None, None

\# Helper method for rounding (needs precision info, e.g., from ConfigManager or ExchangeInfoService)
\# def \_round\_price(self, price: Decimal, trading\_pair: str) \-\> Decimal: ...

### **B. Add Secondary Confirmation Logic (High Priority \- FR-403)**

* **Problem:** Signals are generated based *only* on the prediction threshold.
* **Solution:**
  1. **Access Features:** The \_evaluate\_strategy method needs access to the features that were used to generate the prediction. The PredictionEvent should ideally contain these (associated\_features field).
  2. **Configuration:** Define secondary confirmation rules in the strategy configuration (e.g., confirmation: { "require\_feature": "momentum\_5", "threshold\_gt": 0 }).
  3. **Check Rules:** After the primary prediction threshold is met, iterate through the configured confirmation rules. Check if the required features exist in prediction\_event.associated\_features and if they meet the specified conditions (e.g., greater than, less than threshold).
  4. **Decision:** Only generate the TradeSignalProposedEvent if *both* the primary prediction threshold and *all* secondary confirmation rules pass.

\# In StrategyArbitrator class

async def \_evaluate\_strategy(self, prediction\_event: PredictionEvent) \-\> Optional\[TradeSignalProposedEvent\]:
    \# ... (determine primary side based on prediction\_value) ...

    if side:
        \# \--- Apply Secondary Confirmation \---
        if not self.\_apply\_secondary\_confirmation(prediction\_event, side):
             self.logger.info(f"Primary signal {side} for {prediction\_event.trading\_pair} failed secondary confirmation.", source\_module=self.\_source\_module)
             return None \# Signal rejected by confirmation rules
        \# \--- End Secondary Confirmation \---

        \# \--- If primary and secondary checks pass, proceed \---
        current\_price \= await self.market\_price\_service.get\_latest\_price(...)
        if current\_price is None: return None
        sl\_price, tp\_price \= self.\_calculate\_sl\_tp\_prices(...)
        if sl\_price is None or tp\_price is None: return None
        proposed\_entry \= self.\_determine\_entry\_price(...)

        \# ... (create and return TradeSignalProposedEvent) ...
    \# ...

def \_apply\_secondary\_confirmation(self, prediction\_event: PredictionEvent, primary\_side: str) \-\> bool:
    """Checks if secondary confirmation rules pass."""
    confirmation\_rules \= self.\_mvp\_strategy\_config.get("confirmation\_rules", \[\])
    if not confirmation\_rules:
        return True \# No rules defined, confirmation passes by default

    features \= getattr(prediction\_event, 'associated\_features', None)
    if not features:
        self.logger.warning(f"No associated features found in PredictionEvent {prediction\_event.event\_id} for confirmation.", source\_module=self.\_source\_module)
        return False \# Cannot confirm without features

    for rule in confirmation\_rules:
        feature\_name \= rule.get("feature")
        condition \= rule.get("condition") \# e.g., "gt", "lt", "eq"
        threshold\_str \= rule.get("threshold")

        if not all(\[feature\_name, condition, threshold\_str\]):
            self.logger.warning(f"Skipping invalid confirmation rule: {rule}", source\_module=self.\_source\_module)
            continue

        if feature\_name not in features:
            self.logger.warning(f"Confirmation failed: Required feature '{feature\_name}' not found.", source\_module=self.\_source\_module)
            return False \# Required feature missing

        try:
            feature\_value \= Decimal(str(features\[feature\_name\]))
            threshold \= Decimal(str(threshold\_str))

            passes \= False
            if condition \== "gt" and feature\_value \> threshold: passes \= True
            elif condition \== "lt" and feature\_value \< threshold: passes \= True
            elif condition \== "eq" and feature\_value \== threshold: passes \= True
            \# Add other conditions (gte, lte, ne) if needed

            if not passes:
                 self.logger.info(f"Confirmation failed: Rule {feature\_name} {condition} {threshold} (Value: {feature\_value})", source\_module=self.\_source\_module)
                 return False \# Rule failed

        except (InvalidOperation, TypeError, KeyError) as e:
             self.logger.error(f"Error applying confirmation rule {rule}: {e}", source\_module=self.\_source\_module)
             return False \# Error during rule check

    self.logger.debug("All secondary confirmation rules passed.", source\_module=self.\_source\_module)
    return True \# All rules passed

### **C. Implement Limit Order Entry Price Determination (High Priority)**

* **Problem:** proposed\_entry\_price is None even if entry\_type is "LIMIT".
* **Solution:**
  1. **Get Bid/Ask:** If entry\_type is "LIMIT", fetch the current best bid and ask prices from the MarketPriceService.
  2. **Calculate Limit Price:** Determine the limit price based on the side and current bid/ask. For a BUY, set it at or slightly *below* the current best ask (or based on config offset). For a SELL, set it at or slightly *above* the current best bid.
  3. **Update Event:** Populate proposed\_entry\_price in the TradeSignalProposedEvent.

\# In StrategyArbitrator class

\# Add \_determine\_entry\_price method called in \_evaluate\_strategy
async def \_determine\_entry\_price(self, side: str, current\_price: Decimal) \-\> Optional\[Decimal\]:
     """Determines the proposed entry price based on order type."""
     if self.\_entry\_type \== "MARKET":
          return None \# No specific price for market orders
     elif self.\_entry\_type \== "LIMIT":
          \# Fetch current spread
          spread\_data \= await self.market\_price\_service.get\_bid\_ask\_spread(self.\_current\_trading\_pair) \# Need current pair context
          if spread\_data is None:
               self.logger.warning(f"Cannot determine limit price for {self.\_current\_trading\_pair}: Bid/Ask unavailable. Falling back to current price.", source\_module=self.\_source\_module)
               \# Fallback: Use the current price used for SL/TP calc as limit price
               return current\_price
          else:
               best\_bid, best\_ask \= spread\_data
               \# Configurable offset (e.g., place limit slightly inside spread)
               offset\_pct \= self.\_mvp\_strategy\_config.get("limit\_offset\_pct", Decimal("0.01")) / 100 \# Default 0.01%

               if side \== "BUY":
                    \# Place limit slightly below current ask or at ask
                    limit\_price \= best\_ask \* (Decimal(1) \- offset\_pct)
                    \# Or simply: limit\_price \= best\_ask
               else: \# SELL
                    \# Place limit slightly above current bid or at bid
                    limit\_price \= best\_bid \* (Decimal(1) \+ offset\_pct)
                    \# Or simply: limit\_price \= best\_bid

               \# Add rounding if needed
               \# limit\_price \= self.\_round\_price(limit\_price, self.\_current\_trading\_pair)
               return limit\_price
     else:
          self.logger.error(f"Unsupported entry type for price determination: {self.\_entry\_type}", source\_module=self.\_source\_module)
          return None

\# Ensure \_evaluate\_strategy uses this (see example in Section A)
\# proposed\_entry \= await self.\_determine\_entry\_price(side, current\_price) \# Make \_evaluate\_strategy async

### **D. Improve Error Handling & Validation (Medium Priority)**

* **Problem:** Limited validation of incoming PredictionEvent, inconsistent error handling during init, error messages lack context.
* **Solution:**
  1. **Validate Prediction Event:** Add a \_validate\_prediction\_event helper called in handle\_prediction\_event. Check for required fields (prediction\_value, trading\_pair, etc.), valid types, and reasonable ranges (e.g., probability between 0 and 1). Log warnings and return early if invalid.
  2. **Init Error Handling:** Wrap the \_\_init\_\_ configuration loading in a try...except block. Log detailed errors using self.logger (might need basic logger setup before full config load) and raise a specific custom exception (e.g., StrategyConfigurationError) instead of generic ValueError.
  3. **Contextual Errors:** Add more context (e.g., trading\_pair, event\_id) to error log messages within \_evaluate\_strategy.

### **E. Improve Configuration & Hardcoding (Medium Priority)**

* **Problem:** Assumes binary prediction, hardcoded TP multiplier (in review suggestion), missing config validation.
* **Solution:**
  1. **Prediction Interpretation:** Add configuration (strategy\_arbitrator.prediction\_interpretation: "prob\_up" | "prob\_down" | "price\_change\_pct") to define how prediction\_value should be interpreted. Modify \_evaluate\_strategy to handle these different interpretations when setting prob\_up/prob\_down.
  2. **TP Multiplier:** Make the Risk/Reward ratio for default TP calculation configurable (strategy\_arbitrator.default\_reward\_risk\_ratio).
  3. **Config Validation:** Implement \_validate\_configuration called at the end of \_\_init\_\_. Check ranges for thresholds, SL/TP percentages, valid entry\_type, etc.

### **F. Code Cleanup (Low Priority)**

* **Problem:** Debug print, unused dataclass.
* **Solution:** Remove print("Strategy Arbitrator Loaded"). Remove the unused PredictionPayload dataclass definition.

**Conclusion:** The highest priorities are implementing the core logic for calculating SL/TP and entry prices, and adding secondary confirmation rules. This requires integrating with a market price source. Improving validation, configuration, and error handling will enhance robustness. Addressing the prediction interpretation assumption is also important for flexibility.
