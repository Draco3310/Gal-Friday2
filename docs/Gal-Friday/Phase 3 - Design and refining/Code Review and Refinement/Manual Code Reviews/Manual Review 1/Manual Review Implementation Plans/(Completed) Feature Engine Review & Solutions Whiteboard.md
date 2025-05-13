# **Feature Engine: Review Summary & Solution Whiteboard**

## **1\. Summary of Manual Review Findings (feature\_engine.md)**

The review of src/gal\_friday/feature\_engine.py highlighted:

* **Strengths:** Good event handling, robust error isolation in calculations, appropriate data structures (deque, defaultdict), correct use of Decimal, clear separation of L2/TA features, configurable parameters.
* **Feature Implementation Gaps (High Priority):**
  * Missing Technical Indicators (FR-202): MACD, Bollinger Bands, VWAP, Volatility (ATR/StdDev) are not implemented.
  * Limited Order Book Features (FR-204): Missing Weighted Average Price (WAP), deeper depth analysis beyond basic imbalance.
  * Missing Volume Flow Indicators (FR-205): No implementation found.
* **Code Organization Issues:** Debug print statement, long methods (\_calculate\_l2\_features), potential duplication in TA calculation logic.
* **Error Handling & Robustness:** Limited recovery logic for missing data, inconsistent NaN handling, no validation of incoming market data.
* **Performance Considerations:** Potential overhead from frequent DataFrame conversions, recalculating indicators from scratch instead of incrementally, no memoization for common values (e.g., mid-price).

## **2\. Whiteboard: Proposed Solutions**

Here's a breakdown of solutions addressing the high and medium priority recommendations:

### **A. Implement Missing Features (High Priority \- FR-202, FR-204, FR-205)**

* **Problem:** Key required features (MACD, BBands, VWAP, ATR, WAP, Volume Flow) are missing.
* **Solution:** Add dedicated calculation methods for each missing feature, leveraging pandas-ta where possible for technical indicators and direct calculation for order book features. Integrate these into the \_calculate\_ta\_features and \_calculate\_l2\_features methods or call them separately.
  * **Technical Indicators (Add to \_calculate\_ta\_features or call from there):**
    \# In FeatureEngine class
    import pandas\_ta as ta
    import pandas as pd
    from decimal import Decimal, InvalidOperation

    def \_calculate\_macd\_feature(self, df: pd.DataFrame, trading\_pair: str, interval: str) \-\> dict\[str, str\]:
        """Calculate MACD feature if configured."""
        features \= {}
        cfg \= self.\_feature\_configs.get("macd", {}) \# Get specific config
        fast \= cfg.get("fast\_period", 12\)
        slow \= cfg.get("slow\_period", 26\)
        signal \= cfg.get("signal\_period", 9\)

        if not all(isinstance(p, int) and p \> 0 for p in \[fast, slow, signal\]):
            self.logger.warning(f"Invalid MACD periods configured for {trading\_pair} {interval}", source\_module=self.\_source\_module)
            return features

        feature\_prefix \= f"macd\_{fast}\_{slow}\_{signal}\_{interval}"
        min\_len \= slow \+ signal \-1 \# Approx minimum length needed

        if len(df) \>= min\_len:
            try:
                macd\_df \= ta.macd(df\["close"\], fast=fast, slow=slow, signal=signal)
                if macd\_df is not None and not macd\_df.empty:
                    last\_row \= macd\_df.iloc\[-1\]
                    macd\_line \= last\_row.get(f"MACD\_{fast}\_{slow}\_{signal}")
                    signal\_line \= last\_row.get(f"MACDs\_{fast}\_{slow}\_{signal}")
                    hist \= last\_row.get(f"MACDh\_{fast}\_{slow}\_{signal}")

                    if pd.notna(macd\_line): features\[f"{feature\_prefix}\_line"\] \= f"{macd\_line:.8f}"
                    if pd.notna(signal\_line): features\[f"{feature\_prefix}\_signal"\] \= f"{signal\_line:.8f}"
                    if pd.notna(hist): features\[f"{feature\_prefix}\_hist"\] \= f"{hist:.8f}"
                else:
                     self.logger.debug(f"MACD result is None or empty for {trading\_pair} {interval}", source\_module=self.\_source\_module)

            except Exception as e:
                self.logger.error(f"MACD calculation failed for {trading\_pair} {interval}: {e}", source\_module=self.\_source\_module, exc\_info=True)
        else:
             self.logger.debug(f"Not enough data for MACD ({len(df)} \< {min\_len}) for {trading\_pair} {interval}", source\_module=self.\_source\_module)
        return features

    def \_calculate\_bbands\_feature(self, df: pd.DataFrame, trading\_pair: str, interval: str) \-\> dict\[str, str\]:
        """Calculate Bollinger Bands feature if configured."""
        features \= {}
        cfg \= self.\_feature\_configs.get("bbands", {})
        length \= cfg.get("length", 20\)
        std\_dev \= cfg.get("std\_dev", 2.0)

        if not (isinstance(length, int) and length \> 0 and isinstance(std\_dev, (float, int)) and std\_dev \> 0):
             self.logger.warning(f"Invalid BBands params configured for {trading\_pair} {interval}", source\_module=self.\_source\_module)
             return features

        feature\_prefix \= f"bbands\_{length}\_{std\_dev:.1f}\_{interval}"
        if len(df) \>= length:
            try:
                bbands\_df \= ta.bbands(df\["close"\], length=length, std=std\_dev)
                if bbands\_df is not None and not bbands\_df.empty:
                     last\_row \= bbands\_df.iloc\[-1\]
                     lower \= last\_row.get(f"BBL\_{length}\_{std\_dev}")
                     middle \= last\_row.get(f"BBM\_{length}\_{std\_dev}") \# SMA
                     upper \= last\_row.get(f"BBU\_{length}\_{std\_dev}")
                     bandwidth \= last\_row.get(f"BBB\_{length}\_{std\_dev}") \# Bandwidth
                     percent \= last\_row.get(f"BBP\_{length}\_{std\_dev}") \# %B

                     if pd.notna(lower): features\[f"{feature\_prefix}\_lower"\] \= f"{lower:.8f}"
                     if pd.notna(middle): features\[f"{feature\_prefix}\_middle"\] \= f"{middle:.8f}"
                     if pd.notna(upper): features\[f"{feature\_prefix}\_upper"\] \= f"{upper:.8f}"
                     if pd.notna(bandwidth): features\[f"{feature\_prefix}\_bandwidth"\] \= f"{bandwidth:.6f}"
                     if pd.notna(percent): features\[f"{feature\_prefix}\_percent"\] \= f"{percent:.6f}"
                else:
                     self.logger.debug(f"BBands result is None or empty for {trading\_pair} {interval}", source\_module=self.\_source\_module)
            except Exception as e:
                self.logger.error(f"BBands calculation failed for {trading\_pair} {interval}: {e}", source\_module=self.\_source\_module, exc\_info=True)
        else:
             self.logger.debug(f"Not enough data for BBands ({len(df)} \< {length}) for {trading\_pair} {interval}", source\_module=self.\_source\_module)
        return features

    def \_calculate\_vwap\_feature(self, df: pd.DataFrame, trading\_pair: str, interval: str) \-\> dict\[str, str\]:
         """Calculate VWAP feature (requires high, low, close, volume)."""
         \# Note: VWAP is typically calculated intraday, resetting daily.
         \# Calculating on a rolling window might not be standard VWAP.
         \# pandas-ta vwap might need anchoring ('D' for daily). Check docs.
         \# For simplicity, let's calculate rolling VWAP if configured.
         features \= {}
         cfg \= self.\_feature\_configs.get("vwap", {})
         length \= cfg.get("length", 14\) \# Example: Rolling VWAP over 14 periods

         if not (isinstance(length, int) and length \> 0):
              self.logger.warning(f"Invalid VWAP length configured for {trading\_pair} {interval}", source\_module=self.\_source\_module)
              return features

         feature\_name \= f"vwap\_{length}\_{interval}"
         \# VWAP requires H, L, C, V columns
         required\_cols \= \["high", "low", "close", "volume"\]
         if not all(col in df.columns for col in required\_cols):
              self.logger.warning(f"Missing required columns for VWAP for {trading\_pair} {interval}", source\_module=self.\_source\_module)
              return features

         if len(df) \>= length:
             try:
                 \# Use pandas-ta vwap function
                 vwap\_series \= ta.vwap(df\["high"\], df\["low"\], df\["close"\], df\["volume"\], length=length)
                 if vwap\_series is not None and not vwap\_series.empty:
                     last\_vwap \= vwap\_series.iloc\[-1\]
                     if pd.notna(last\_vwap):
                         features\[feature\_name\] \= f"{last\_vwap:.8f}"
                     else:
                          self.logger.debug(f"VWAP is NaN for {trading\_pair} {interval}", source\_module=self.\_source\_module)
                 else:
                      self.logger.debug(f"VWAP result is None or empty for {trading\_pair} {interval}", source\_module=self.\_source\_module)
             except Exception as e:
                 self.logger.error(f"VWAP calculation failed for {trading\_pair} {interval}: {e}", source\_module=self.\_source\_module, exc\_info=True)
         else:
              self.logger.debug(f"Not enough data for VWAP ({len(df)} \< {length}) for {trading\_pair} {interval}", source\_module=self.\_source\_module)
         return features

    def \_calculate\_volatility\_feature(self, df: pd.DataFrame, trading\_pair: str, interval: str) \-\> dict\[str, str\]:
        """Calculate ATR and/or Stdev volatility features if configured."""
        features \= {}
        atr\_cfg \= self.\_feature\_configs.get("atr", {})
        stdev\_cfg \= self.\_feature\_configs.get("stdev", {})
        atr\_len \= atr\_cfg.get("length", 14\)
        stdev\_len \= stdev\_cfg.get("length", 14\)

        \# ATR Calculation
        if isinstance(atr\_len, int) and atr\_len \> 0:
             feature\_name \= f"atr\_{atr\_len}\_{interval}"
             required\_cols \= \["high", "low", "close"\]
             if all(col in df.columns for col in required\_cols) and len(df) \>= atr\_len:
                  try:
                       atr\_series \= ta.atr(df\["high"\], df\["low"\], df\["close"\], length=atr\_len)
                       if atr\_series is not None and not atr\_series.empty:
                            last\_atr \= atr\_series.iloc\[-1\]
                            if pd.notna(last\_atr): features\[feature\_name\] \= f"{last\_atr:.8f}"
                       else: self.logger.debug(f"ATR result None/empty for {trading\_pair} {interval}", source\_module=self.\_source\_module)
                  except Exception as e: self.logger.error(f"ATR calc failed for {trading\_pair} {interval}: {e}", source\_module=self.\_source\_module, exc\_info=True)
             else: self.logger.debug(f"Not enough data/cols for ATR ({len(df)} \< {atr\_len}) for {trading\_pair} {interval}", source\_module=self.\_source\_module)

        \# Standard Deviation Calculation
        if isinstance(stdev\_len, int) and stdev\_len \> 0:
             feature\_name \= f"stdev\_{stdev\_len}\_{interval}"
             if "close" in df.columns and len(df) \>= stdev\_len:
                  try:
                       stdev\_series \= ta.stdev(df\["close"\], length=stdev\_len)
                       if stdev\_series is not None and not stdev\_series.empty:
                            last\_stdev \= stdev\_series.iloc\[-1\]
                            if pd.notna(last\_stdev): features\[feature\_name\] \= f"{last\_stdev:.8f}"
                       else: self.logger.debug(f"Stdev result None/empty for {trading\_pair} {interval}", source\_module=self.\_source\_module)
                  except Exception as e: self.logger.error(f"Stdev calc failed for {trading\_pair} {interval}: {e}", source\_module=self.\_source\_module, exc\_info=True)
             else: self.logger.debug(f"Not enough data/cols for Stdev ({len(df)} \< {stdev\_len}) for {trading\_pair} {interval}", source\_module=self.\_source\_module)

        return features

    \# Update \_calculate\_ta\_features to call the new methods
    def \_calculate\_ta\_features(self, trading\_pair: str, interval: str) \-\> dict\[str, str\] | None:
        \# ... (prepare df as before) ...
        if df is None: return None
        features \= {}
        features.update(self.\_calculate\_rsi\_feature(df, trading\_pair, interval))
        features.update(self.\_calculate\_roc\_feature(df, trading\_pair, interval))
        features.update(self.\_calculate\_macd\_feature(df, trading\_pair, interval)) \# Added
        features.update(self.\_calculate\_bbands\_feature(df, trading\_pair, interval)) \# Added
        features.update(self.\_calculate\_vwap\_feature(df, trading\_pair, interval)) \# Added
        features.update(self.\_calculate\_volatility\_feature(df, trading\_pair, interval)) \# Added
        \# ... Add calls for other TAs ...
        return features if features else None

  * **Order Book Features (Add to \_calculate\_l2\_features):**
    \# In \_calculate\_l2\_features method, after basic spread/mid-price

        \# \--- Weighted Average Price (WAP) & Deeper Depth \---
        depth\_levels \= self.\_feature\_configs.get("l2\_depth\_levels", \[1, 5, 10\]) \# Configurable levels

        for level in depth\_levels:
            if not isinstance(level, int) or level \<= 0: continue

            try:
                \# Cumulative Volume at depth
                bid\_vol\_cum \= sum(Decimal(b\[1\]) for b in bids\[:level\])
                ask\_vol\_cum \= sum(Decimal(a\[1\]) for a in asks\[:level\])
                features\[f"bid\_vol\_cum\_{level}"\] \= f"{bid\_vol\_cum:.8f}"
                features\[f"ask\_vol\_cum\_{level}"\] \= f"{ask\_vol\_cum:.8f}"

                \# Weighted Average Price (WAP) at depth
                if bid\_vol\_cum \> Decimal(0):
                    bid\_wap \= sum(Decimal(b\[0\]) \* Decimal(b\[1\]) for b in bids\[:level\]) / bid\_vol\_cum
                    features\[f"bid\_wap\_{level}"\] \= f"{bid\_wap:.8f}"
                if ask\_vol\_cum \> Decimal(0):
                    ask\_wap \= sum(Decimal(a\[0\]) \* Decimal(a\[1\]) for a in asks\[:level\]) / ask\_vol\_cum
                    features\[f"ask\_wap\_{level}"\] \= f"{ask\_wap:.8f}"

            except (IndexError, ValueError, InvalidOperation) as calc\_error:
                 self.logger.error(f"Error calculating depth/WAP features at level {level} for {trading\_pair}: {calc\_error}", source\_module=self.\_source\_module)
            except Exception as depth\_error:
                 self.logger.error(f"Unexpected error calculating depth/WAP features at level {level} for {trading\_pair}: {depth\_error}", source\_module=self.\_source\_module, exc\_info=True)

        \# \--- Add other L2 features here (e.g., price level density) \---

  * **Volume Flow Indicators (FR-205):** This requires access to *trade* data (not just OHLCV or L2 book updates), which the current FeatureEngine doesn't seem to consume.
    * **Action:** Clarify if trade data events (MarketDataTradeEvent?) will be provided. If so, subscribe to them and implement indicators like Volume Weighted Average Price (VWAP \- potentially more accurate with trades), Order Flow Imbalance (requires tracking aggressor side of trades), etc. If not, FR-205 cannot be met by this module alone.

### **B. Code Organization (High/Medium Priority)**

* **Problem:** Debug print statement, long methods, duplicate TA logic.
* **Solution:**
  1. **Remove Print:** Replace print("Feature Engine Loaded") with self.logger.info("Feature Engine module loaded", source\_module=\_\_name\_\_) (or similar) inside \_\_init\_\_.
  2. **Refactor \_calculate\_l2\_features:** Break it down into smaller helper methods for calculating basic stats (bid/ask/mid/spread), imbalance, WAP/depth, etc.
  3. **Refactor TA Methods:** Create a generic helper function \_calculate\_ta\_indicator that takes the DataFrame, indicator name (e.g., "rsi", "macd"), parameters, and the specific pandas-ta function (or a lambda wrapper) as arguments. This helper would handle the common logic (checking data length, calling pandas-ta, handling NaNs, error logging). The specific methods (\_calculate\_rsi\_feature, etc.) would become thin wrappers calling this generic helper. (See review suggestion for an example structure).

### **C. Error Handling & Robustness (Medium Priority)**

* **Problem:** Limited recovery for missing data, inconsistent NaN handling, no input validation.
* **Solution:**
  1. **Input Validation:** Add validation checks at the beginning of \_handle\_l2\_event and \_handle\_ohlcv\_event to ensure the incoming event data has the expected structure and types before processing. Check for non-numeric values, negative prices/volumes in OHLCV, etc. Log warnings and skip processing invalid events. (See review suggestion for \_validate\_ohlcv\_data).
  2. **NaN Handling:** Standardize how NaNs from pandas-ta are handled. Decide whether to publish features with NaN (represented as empty string or specific marker?) or simply omit the feature if NaN occurs. Log consistently (e.g., always DEBUG level if NaN).
  3. **Data Availability:** Add checks within calculation methods to see how old the latest L2 or OHLCV data is. If data is too stale (beyond a configurable threshold), avoid calculating features that depend on it or publish features with a specific "stale\_data" flag/value.

### **D. Performance Considerations (Medium Priority)**

* **Problem:** Potential bottlenecks due to DataFrame conversion, full recalculations, lack of memoization.
* **Solution:**
  1. **DataFrame Optimization:** Implement DataFrame caching as suggested in the review (\_get\_cached\_dataframe). Only recreate the DataFrame if the underlying deque has changed size. *Alternative:* Explore libraries that calculate TAs directly on deques/lists if pandas-ta overhead proves significant.
  2. **Incremental Calculation:** This is harder with pandas-ta which often expects the full series. For some indicators (like SMA, EMA), manual incremental updates are possible but complex to implement correctly. Focus on DataFrame caching first.
  3. **Memoization:** Use @functools.lru\_cache(maxsize=...) or a simple dictionary cache for frequently recalculated values derived from L2 data within a single event processing cycle (like mid-price, spread) if \_calculate\_l2\_features is refactored into smaller pieces. (See review suggestion for \_get\_mid\_price).

Addressing the feature gaps (A) is crucial for meeting functional requirements. Improving code organization (B), robustness (C), and performance (D) will enhance maintainability and reliability.
