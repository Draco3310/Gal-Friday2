# **Core Events (events.py): Review Summary & Solution Whiteboard**

## **1\. Summary of Manual Review Findings (Events.md)**

The review of src/gal\_friday/core/events.py found the following:

* **Strengths:** Implements all required event types, uses frozen dataclasses for immutability, good docstrings and type hints, logical organization, uses Decimal appropriately for financial values, consistent UUID usage. Technically sound with no syntax/type errors.
* **Type Inconsistency:**
  * The inter-module communication specification document often defines numeric values (prices, quantities) as strings.
  * The events.py implementation correctly uses Decimal for most financial values (good practice) but uses strings for OHLCV values (matching the spec). This inconsistency needs resolution.
* **Missing Fields:**
  * TradeSignalApprovedEvent: Missing the risk\_parameters: Dict\[str, Any\] field specified in the spec.
  * TradeSignalProposedEvent: Uses triggering\_prediction\_event\_id: Optional\[uuid.UUID\] (a reference) instead of including the full triggering\_prediction: Dict\[str, Any\] (the prediction event data) as specified in the spec.
* **Validation:** Lacks explicit validation for field values (e.g., ensuring prices/quantities are positive, sides are "BUY" or "SELL").
* **Documentation/Cleanup:** While comments were cleaned up, documentation explaining design choices (like Decimal vs. String) is missing.

## **2\. Whiteboard: Proposed Solutions**

Here's a breakdown of solutions addressing the high and medium priority recommendations:

### **A. Harmonize Types (High Priority)**

* **Problem:** Discrepancy between specification (strings for numbers) and implementation (mostly Decimal, except OHLCV).
* **Solution:**
  1. **Decision:** The implementation's use of Decimal for financial calculations (TradeSignalApprovedEvent, ExecutionReportEvent, etc.) is generally **preferable** to strings for preventing precision errors. OHLCV data often comes as strings from exchanges, so keeping MarketDataOHLCVEvent fields (open, high, low, close, volume) as strings is acceptable *at that stage*, assuming downstream consumers (like FeatureEngine) will parse them into Decimal or float as needed. The MarketDataL2Event also uses strings, likely reflecting the raw exchange data format needed for checksums.
  2. **Action:**
     * **Update Specification Document:** Modify the inter-module communication document to reflect the use of Decimal for relevant fields in events like TradeSignalApprovedEvent, TradeSignalProposedEvent, and ExecutionReportEvent. Explicitly state that MarketDataL2Event and MarketDataOHLCVEvent use strings to represent the raw exchange data format.
     * **Verify Consistency:** Double-check all event classes in events.py to ensure the chosen types (Decimal or str) are used consistently according to the updated decision.

### **B. Add Missing Fields (High Priority)**

* **Problem:** Two event classes are missing fields defined in the specification.
* **Solution:**
  1. **TradeSignalApprovedEvent:** Add the risk\_parameters field.
     \# In events.py
     from typing import Any, Dict \# Ensure Any, Dict are imported

     @dataclass(frozen=True)
     class TradeSignalApprovedEvent(Event):
         \# ... existing fields ...
         quantity: Decimal
         sl\_price: Decimal
         tp\_price: Decimal
         limit\_price: Optional\[Decimal\] \= None
         \# \--- ADDED FIELD \---
         risk\_parameters: Dict\[str, Any\] \# Parameters used by RiskManager for approval
         \# \--- END ADDED FIELD \---
         event\_type: EventType \= field(default=EventType.TRADE\_SIGNAL\_APPROVED, init=False)

  2. **TradeSignalProposedEvent:** Address the triggering\_prediction discrepancy.
     * **Option 1 (Add Full Data \- Matches Spec):** Add the triggering\_prediction dictionary. This makes the event self-contained but potentially larger.
       \# In events.py

       @dataclass(frozen=True)
       class TradeSignalProposedEvent(Event):
           \# ... existing fields ...
           strategy\_id: str
           proposed\_entry\_price: Optional\[Decimal\] \= None
           \# \--- REVISED FIELD \---
           \# Keep ID for tracing if desired
           triggering\_prediction\_event\_id: Optional\[uuid.UUID\] \= None
           \# Add the actual prediction data as per spec (or a subset)
           triggering\_prediction: Optional\[Dict\[str, Any\]\] \= None
           \# \--- END REVISED FIELD \---
           event\_type: EventType \= field(default=EventType.TRADE\_SIGNAL\_PROPOSED, init=False)

     * **Option 2 (Keep Reference \- Current):** Keep only the triggering\_prediction\_event\_id. This is more lightweight but requires the consumer (e.g., RiskManager) to potentially fetch the original PredictionEvent if it needs the full data.
     * **Decision & Action:** Discuss which approach is better. Adding the full data makes the event more complete but increases size. Keeping the reference is efficient if consumers rarely need the full prediction details. **Recommendation:** Start by adding the full triggering\_prediction dictionary (Option 1\) to align with the spec unless there's a strong performance reason not to. Document this choice.

### **C. Add Validation (Medium Priority)**

* **Problem:** No validation on critical event fields (e.g., ensuring side is "BUY" or "SELL", prices/quantities are positive).
* **Solution:** Implement validation logic. Since dataclasses are frozen, validation should ideally happen *before* instantiation. Factory methods are a good place for this. Alternatively, add a \_\_post\_init\_\_ method (though modifying frozen dataclasses post-init is tricky; it's better for validation checks that raise errors). Using a library like Pydantic instead of standard dataclasses would offer built-in validation.
  * **Approach 1: Validation in Factory Methods (Recommended if keeping dataclasses):**
    \# In events.py
    from decimal import InvalidOperation

    @dataclass(frozen=True)
    class TradeSignalApprovedEvent(Event):
        \# ... fields ...

        @classmethod
        def create(
            cls,
            source\_module: str,
            signal\_id: uuid.UUID,
            trading\_pair: str,
            exchange: str,
            side: str,
            order\_type: str,
            quantity: Decimal,
            sl\_price: Decimal,
            tp\_price: Decimal,
            risk\_parameters: Dict\[str, Any\], \# Added field
            limit\_price: Optional\[Decimal\] \= None,
            \# Add other necessary args...
        ) \-\> "TradeSignalApprovedEvent":
            \# \--- VALIDATION \---
            if side not in \["BUY", "SELL"\]:
                raise ValueError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'.")
            if order\_type not in \["LIMIT", "MARKET"\]:
                raise ValueError(f"Invalid order\_type: {order\_type}. Must be 'LIMIT' or 'MARKET'.")
            if order\_type \== "LIMIT" and limit\_price is None:
                raise ValueError("limit\_price must be provided for LIMIT orders.")
            if order\_type \== "MARKET" and limit\_price is not None:
                \# Or just ignore limit\_price for MARKET? Decide policy.
                raise ValueError("limit\_price should not be provided for MARKET orders.")
            if quantity \<= Decimal(0):
                raise ValueError(f"Quantity must be positive: {quantity}")
            if sl\_price \<= Decimal(0):
                raise ValueError(f"Stop loss price must be positive: {sl\_price}")
            if tp\_price \<= Decimal(0):
                raise ValueError(f"Take profit price must be positive: {tp\_price}")
            if limit\_price is not None and limit\_price \<= Decimal(0):
                 raise ValueError(f"Limit price must be positive: {limit\_price}")
            \# Add more checks (e.g., SL/TP relative to side/price)
            \# \--- END VALIDATION \---

            \# Create and return instance
            return cls(
                source\_module=source\_module,
                event\_id=uuid.uuid4(),
                timestamp=datetime.utcnow(),
                signal\_id=signal\_id,
                trading\_pair=trading\_pair,
                exchange=exchange,
                side=side,
                order\_type=order\_type,
                quantity=quantity,
                sl\_price=sl\_price,
                tp\_price=tp\_price,
                limit\_price=limit\_price,
                risk\_parameters=risk\_parameters, \# Added field
            )

  * **Approach 2: Consider Pydantic:** If validation becomes complex or widespread, migrating to Pydantic models could simplify it significantly.

### **D. Add Factory Methods (Medium Priority)**

* **Problem:** Instantiating events requires manually setting source\_module, event\_id, and timestamp every time. This is repetitive and error-prone.
* **Solution:** Implement class methods (factories) like .create(...) for each event type. These methods handle the creation of metadata (event\_id, timestamp) automatically and can incorporate validation (as shown in section C).
  * **Action:** Add .create(...) methods to all relevant event classes in events.py. Ensure they take the specific payload fields as arguments and handle the base Event fields internally. (See example in section C).

### **E. Add Documentation (Medium Priority)**

* **Problem:** Lack of documentation explaining *why* certain design choices were made (e.g., Decimal vs. String).
* **Solution:** Add module-level docstrings or comments within the code explaining the rationale behind key decisions, especially the type choices for financial data and the handling of the triggering\_prediction field.
  \# In events.py (Module level or near relevant classes)

  """
  Core Event Definitions for Gal-Friday

  Design Notes:
  \- Events are implemented as frozen dataclasses for immutability.
  \- Financial values (prices, quantities, commissions) in trade-related events
    (signals, execution reports) use decimal.Decimal for precision.
  \- Market data events (L2, OHLCV) use strings for numeric fields to accurately
    represent the raw data format received from exchanges, which may be needed
    for checksums or initial parsing consistency. Downstream modules are
    responsible for converting these strings to numeric types (Decimal/float)
    as needed for calculations.
  \- Timestamps should generally be UTC. Exchange timestamps are included where available.
  \- UUIDs are used for unique event identification and signal correlation.
  """

By addressing these points, particularly harmonizing types and adding missing fields, the events.py module will be more robust and consistent with the system's specifications. Adding validation and factory methods will further improve usability and data integrity.
