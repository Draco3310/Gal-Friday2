# **Simulated Market Price Service (simulated\_market\_price\_service.py): Review Summary & Solution Whiteboard**

## **1\. Summary of Manual Review Findings (simulated\_market\_price\_service.md)**

* **Strengths:** Clean basic interface (get\_latest\_price), good time progression handling (update\_time), flexible price lookup (asof), handles self-conversion (e.g., USD/USD).
* **Functional Requirements Gaps (High Priority):**
  * Missing Bid/Ask Spread Simulation: Only provides a single 'close' price.
  * No Market Depth Simulation: Cannot simulate order book for realistic slippage.
  * Limited Price Data Provided: Only 'close' price accessible via the main method.
  * Incomplete Interface: Doesn't implement the full expected MarketPriceService interface (e.g., missing get\_bid\_ask\_spread, lifecycle methods start/stop, freshness checks).
* **Design & Implementation Issues:**
  * Module-level logger used instead of injected LoggerService.
  * Inconsistent error handling (returns None silently).
  * Interface alignment with the *real* MarketPriceService is unclear/missing.
  * Lacks clear integration points for backtesting engine control beyond update\_time.
  * Uses synchronous methods (get\_latest\_price) where the abstract interface likely expects asynchronous (async def).
* **Error Handling Concerns:** Minimal validation of input historical data, silent None returns on error, no validation of update\_time input.
* **Configuration & Hardcoding:** No configuration for simulation parameters (spread, price column); hardcoded 'close' column usage.

## **2\. Whiteboard: Proposed Solutions**

The main goals are to make the simulation more realistic (spread, depth) and align the class with the MarketPriceService abstract interface.

### **A. Implement Bid/Ask Spread Simulation (High Priority)**

* **Problem:** Only a single price ('close') is returned, not a realistic bid/ask spread.
* **Solution:**
  1. **Add get\_bid\_ask Method:** Implement the async def get\_bid\_ask\_spread(self, trading\_pair: str) \-\> Optional\[Tuple\[Decimal, Decimal\]\] method as required by the (updated) MarketPriceService ABC.
  2. **Spread Calculation:** Inside get\_bid\_ask\_spread:
     * Get the base 'close' price using get\_latest\_price.
     * If the price exists, calculate a spread based on a configurable model:
       * **Simple Percentage:** half\_spread \= close\_price \* (spread\_pct / 200). spread\_pct comes from config (simulation.spread.default\_pct or pair-specific).
       * **Volatility-Adjusted (Better):** Calculate recent volatility (e.g., ATR or std dev of returns from historical data up to \_current\_timestamp). Adjust the base spread percentage based on volatility: adjusted\_spread\_pct \= base\_spread\_pct \* (1 \+ volatility \* volatility\_multiplier). volatility\_multiplier comes from config.
     * Calculate bid \= close\_price \- half\_spread and ask \= close\_price \+ half\_spread.
     * Return (bid, ask) or None if the base price is unavailable or spread calculation fails. Ensure bid \< ask.

\# In SimulatedMarketPriceService class

\# Add in \_\_init\_\_ (assuming injected config\_manager and logger\_service):
\# self.config \= config\_manager
\# self.logger \= logger\_service
\# self.\_load\_simulation\_config() \# Load spread params etc.

\# Method to load config
\# def \_load\_simulation\_config(self):
\#     sim\_config \= self.config.get("simulation", {})
\#     self.\_default\_spread\_pct \= self.config.get\_decimal("simulation.spread.default\_pct", Decimal("0.1"))
\#     self.\_spread\_config \= sim\_config.get("spread", {}) \# Pair specific overrides
\#     self.\_volatility\_multiplier \= self.config.get\_decimal("simulation.spread.volatility\_multiplier", Decimal("1.5"))
\#     \# ... other sim params ...

\# Implement the required async method
async def get\_bid\_ask\_spread(self, trading\_pair: str) \-\> Optional\[Tuple\[Decimal, Decimal\]\]:
    """Gets the simulated bid and ask prices at the current simulation time."""
    \# Note: Even though simulation logic might be fundamentally synchronous based on
    \# the current time, the method signature MUST be async to match the ABC.
    \# We don't actually need to await anything internal here unless fetching
    \# volatility itself becomes async.

    \# Use internal synchronous method to get base price
    close\_price \= self.get\_latest\_price(trading\_pair) \# This is the existing sync method
    if close\_price is None:
        return None

    try:
        \# Get spread config
        pair\_spread\_cfg \= self.\_spread\_config.get(trading\_pair, {})
        base\_spread\_pct \= Decimal(str(pair\_spread\_cfg.get("base\_pct", self.\_default\_spread\_pct)))

        \# \--- Optional: Adjust spread for volatility \---
        \# Requires a method to calculate volatility based on self.\_current\_timestamp
        \# volatility \= self.\_calculate\_volatility(trading\_pair) \# Sync or async?
        \# if volatility is not None:
        \#     volatility\_factor \= Decimal(1) \+ (volatility \* self.\_volatility\_multiplier / Decimal(100))
        \#     spread\_pct \= base\_spread\_pct \* volatility\_factor
        \# else:
        \#     spread\_pct \= base\_spread\_pct
        \# spread\_pct \= min(spread\_pct, Decimal("2.0")) \# Cap spread
        \# \--- End Volatility Adjustment \---

        \# Simple percentage spread for now:
        spread\_pct \= base\_spread\_pct

        if spread\_pct \< 0: spread\_pct \= Decimal(0) \# Ensure non-negative

        half\_spread\_amount \= close\_price \* (spread\_pct / Decimal(200))

        \# Ensure precision
        \# context \= decimal.Context(prec=self.config.get\_int("decimal\_precision", 28))
        \# bid \= context.subtract(close\_price, half\_spread\_amount)
        \# ask \= context.add(close\_price, half\_spread\_amount)
        bid \= close\_price \- half\_spread\_amount
        ask \= close\_price \+ half\_spread\_amount

        \# Ensure bid \< ask, handle potential zero spread
        if bid \>= ask:
             \# If spread is tiny/zero, create minimal spread based on price precision
             \# Needs precision info for the pair
             \# min\_tick \= self.\_get\_min\_tick(trading\_pair) \# Needs helper
             \# bid \= close\_price \- min\_tick / 2
             \# ask \= close\_price \+ min\_tick / 2
             \# Fallback: return None or log warning if bid \>= ask
             if bid \== ask: \# Zero spread case
                  \# Could return identical bid/ask or None
                  self.logger.debug(f"Calculated zero spread for {trading\_pair} at {close\_price}", source\_module=self.\_source\_module)
                  \# return (bid, ask) \# Option 1: Return zero spread
                  return None \# Option 2: Indicate invalid spread
             else: \# Crossed book simulation? Unlikely here.
                  self.logger.warning(f"Simulated spread resulted in bid \>= ask for {trading\_pair}", source\_module=self.\_source\_module)
                  return None

        return (bid, ask)

    except Exception as e:
        self.logger.error(f"Error calculating simulated spread for {trading\_pair}: {e}", exc\_info=True, source\_module=self.\_source\_module)
        return None

\# Need helper methods like \_calculate\_volatility, \_get\_min\_tick if used

### **B. Align Interface with Real Service (High Priority)**

* **Problem:** Missing methods (start, stop, get\_price\_timestamp, is\_price\_fresh) required by the MarketPriceService ABC. Existing get\_latest\_price is synchronous.
* **Solution:**
  1. **Add Missing Methods:** Add the required async def methods to SimulatedMarketPriceService matching the ABC signatures.
  2. **Implement Logic:**
     * start/stop: Can have minimal logic (e.g., logging) as there are no connections to manage, but they must exist and be async.
     * get\_price\_timestamp: Return self.\_current\_timestamp if the price lookup for the pair at that time was successful.
     * is\_price\_fresh: Check if self.\_current\_timestamp is not None and if data exists for the pair at or before that time using asof. The concept of "freshness" relative to real-time doesn't directly apply, but it can indicate if data *is available* at the current simulation time.
  3. **Adapt Existing Method:** Rename the current synchronous get\_latest\_price to something like \_get\_latest\_price\_at\_current\_time (private). Implement the required async def get\_latest\_price which simply calls the internal synchronous method (no real await needed, but signature matches ABC).

\# In SimulatedMarketPriceService class
from datetime import datetime, timezone \# Add timezone

\# Rename existing synchronous method
def \_get\_latest\_price\_at\_current\_time(self, trading\_pair: str) \-\> Optional\[Decimal\]:
    \# ... (existing synchronous logic using self.\_current\_timestamp) ...
    \# ... (returns Decimal or None) ...
    \# Keep this method synchronous as it accesses pandas data based on pre-set time

\# Implement required async methods matching ABC

async def start(self) \-\> None:
    """Initializes the simulated service (no-op for simulation)."""
    self.logger.info("SimulatedMarketPriceService started.", source\_module=self.\_source\_module)
    \# No external connections needed

async def stop(self) \-\> None:
    """Stops the simulated service (no-op for simulation)."""
    self.logger.info("SimulatedMarketPriceService stopped.", source\_module=self.\_source\_module)
    \# No external connections

\# Implement async interface method by calling the synchronous logic
async def get\_latest\_price(self, trading\_pair: str) \-\> Optional\[Decimal\]:
    """Gets the latest known price at the current simulation time."""
    \# No actual await needed here, but signature matches ABC
    return self.\_get\_latest\_price\_at\_current\_time(trading\_pair)

\# get\_bid\_ask\_spread (already added above, ensure it's async def)

async def get\_price\_timestamp(self, trading\_pair: str) \-\> Optional\[datetime\]:
     """Gets the simulation timestamp for which the current price is valid."""
     \# Check if data exists for the pair at the current time
     price \= self.\_get\_latest\_price\_at\_current\_time(trading\_pair)
     if price is not None:
          \# Return the current simulation time if price lookup was successful
          return self.\_current\_timestamp
     else:
          \# Price lookup failed (no data at or before \_current\_timestamp)
          return None

async def is\_price\_fresh(self, trading\_pair: str, max\_age\_seconds: float \= 60.0) \-\> bool:
     """Checks if price data is available at the current simulation time."""
     \# In simulation, "freshness" means data exists for the current timestamp.
     \# The max\_age\_seconds parameter isn't directly relevant here.
     price\_ts \= await self.get\_price\_timestamp(trading\_pair)
     return price\_ts is not None \# True if data was found for the current sim time

### **C. Implement Market Depth Simulation (High Priority)**

* **Problem:** No order book depth provided, preventing realistic slippage simulation based on size.
* **Solution:** Implement get\_order\_book\_snapshot (or similar method if defined in ABC).
  * Get the current simulated bid and ask from get\_bid\_ask\_spread.
  * Generate plausible price levels around the bid/ask (e.g., decreasing/increasing by a small percentage or tick size).
  * Generate plausible volume for each level (e.g., starting with a base volume from config and decaying for levels further from the BBO).
  * Return a dictionary {'bids': \[\[price\_str, vol\_str\], ...\], 'asks': \[\[price\_str, vol\_str\], ...\]}.

*(See code example in review document recommendation)*

### **D. Replace Logger & Improve Error Handling/Validation (Medium Priority)**

* **Problem:** Uses module-level logger; inconsistent error handling; minimal data validation.
* **Solution:**
  1. **Inject Logger:** Modify \_\_init\_\_ to accept logger\_service: LoggerService and assign it to self.logger. Replace all log.\* calls with self.logger.\*.
  2. **Error Handling:** Standardize error handling. Instead of just returning None, log specific error messages (e.g., "No historical data for pair", "Simulation time not set", "Timestamp before data start"). Return None consistently on failure.
  3. **Validation:**
     * In \_\_init\_\_, add more thorough checks on the input historical\_data DataFrames (e.g., check for required columns like 'open', 'high', 'low', 'close', 'volume' if simulating more than just close; check index is sorted).
     * In update\_time, validate the input timestamp is a datetime object.

### **E. Add Configuration (Medium Priority)**

* **Problem:** Hardcoded 'close' column, no config for simulation parameters (spread, depth).
* **Solution:**
  1. Inject ConfigManager into \_\_init\_\_.
  2. Load parameters like default spread percentage, volatility multipliers, base volume for depth simulation, price column to use (e.g., 'close', 'open') from config in \_\_init\_\_ or a dedicated \_load\_configuration method.
  3. Use these configured values in the simulation logic (e.g., self.\_price\_column, self.\_default\_spread\_pct).

**Conclusion:** The SimulatedMarketPriceService needs significant enhancement to be a realistic simulation tool and to comply with the MarketPriceService interface. Implementing bid/ask spread, market depth, and aligning the async interface are the highest priorities. Adding configurability and improving logging/error handling are also important.
