# **Historical Data Service: Review Summary & Solution Whiteboard (Updated)**

## **1\. Summary of Manual Review Findings (historical\_data\_service.md)**

*(Summary remains the same as the previous version \- highlighting incomplete implementation, lack of validation, missing rate limits, etc.)*

## **1.5 Context from Supporting Documents**

* **SRS (srs\_gal\_friday\_v0.1.md):**
  * Confirms the need for historical data primarily for **Backtesting** (FR-1001) and **Model Retraining** (FR-309).
  * Specifies interaction with **Kraken REST API** for fetching historical data (NFR-301).
  * Requires storage in **PostgreSQL** (FR-806 for critical data) and **InfluxDB** (FR-807 for time-series).
* **Architecture Concept (architecture\_concept\_gal\_friday\_v0.1.md):**
  * Doesn't explicitly list HistoricalDataService as a top-level module, but implies its functionality is needed by the BacktestingEngine (Section 3, Point 10\) and potentially the PredictionService for retraining (Section 4, ML inference offloading implies data access). Its role is likely that of a support service or part of the backtesting/retraining infrastructure.
* **Interface Definitions (interface\_definitions\_gal\_friday\_v0.1.md):**
  * Does *not* define an interface for HistoricalDataService. The provided historical\_data\_service.py file likely serves as this definition (an Abstract Base Class).
* **DB Schema (db\_schema\_gal\_friday\_v0.1.md):**
  * Explicitly defines where historical data should be stored:
    * **InfluxDB Measurement market\_data\_ohlcv:** Intended for time-series OHLCV data (Section 3.2). This aligns with the need for efficient time-based querying for backtesting and feature generation during retraining. Tags include exchange, trading\_pair, interval. Fields include open, high, low, close, volume (as Floats).
    * *(No specific PostgreSQL table defined for historical OHLCV, suggesting InfluxDB is the primary target for this specific data type as per the schema doc).*
* **Inter-module Communication (inter\_module\_comm\_gal\_friday\_v0.1.md):**
  * No specific events or APIs are defined for *publishing* historical data, reinforcing its role as a service queried by others (like Backtester) rather than an active event publisher.
* **Project Plan (project\_plan\_gal\_friday\_v0.1.md):**
  * Does not explicitly list a task for creating the HistoricalDataService itself, but tasks like "Implement Basic Backtesting Engine" (3.14) and "Implement Model Training/Loading Script" (3.15) implicitly depend on the ability to access historical data.

## **2\. Whiteboard: Proposed Solutions (for the Concrete Implementation)**

*(Based on the review document and refined by the supporting documents)*

### **A. Implement Robust Data Validation (High Priority)**

* **Problem:** Data retrieved from external APIs isn't sufficiently validated.
* **Solution:** Implement \_validate\_ohlcv\_data (as detailed previously). This is crucial before storing data in InfluxDB to ensure data quality for backtesting and retraining (implicit requirement from FR-1001, FR-309).

### **B. Add Proper Rate Limit Handling (High Priority)**

* **Problem:** Fetching historical data often involves many sequential API calls, making rate limiting essential.
* **Solution:** Implement RateLimitTracker and use await rate\_limiter.wait\_if\_needed() before *each* API call made to fetch historical data chunks (e.g., within \_fetch\_ohlcv\_data). This aligns with NFR-301 (interfacing with Kraken REST API) and avoids violating Kraken's terms.

### **C. Implement Database Storage (High Priority)**

* **Problem:** Persistence layer is incomplete.
* **Solution:**
  1. **Storage Target:** Based on the DB Schema (Section 3.2), the primary target for historical OHLCV data is **InfluxDB**. The concrete implementation should focus on interacting with InfluxDB. (PostgreSQL might store *metadata* about the historical data ranges, but the time-series data itself goes to InfluxDB according to the schema doc).
  2. **InfluxDB Interaction:** Use the influxdb-client-python library.
  3. **Schema Alignment:** Ensure data is written to the market\_data\_ohlcv measurement with tags (exchange, trading\_pair, interval) and fields (open, high, low, close, volume as Floats) as defined in the DB Schema doc. The timestamp should be the bar's start time.
  4. **Implement \_store\_ohlcv\_data\_in\_influxdb:** Write the logic to batch-write data points to InfluxDB for efficiency. Handle potential connection errors and write failures.
  5. **Implement \_query\_ohlcv\_data\_from\_influxdb:** Write the logic to query InfluxDB using Flux queries based on measurement, tags (pair, interval), and time range. Return data likely as a Pandas DataFrame, aligning with the ABC interface (get\_historical\_ohlcv).

### **D. Implement Incremental Updates (Medium Priority)**

* **Problem:** Requires full re-downloads.
* **Solution:**
  1. **Get Last Timestamp:** Implement \_get\_latest\_timestamp\_from\_influxdb to query InfluxDB for the last timestamp for a given measurement/tag set.
  2. **Modify Fetch Logic:** Fetch only data *after* the last stored timestamp (plus buffer) using the API's since parameter.
  3. **Store New Data:** Store the newly fetched (and validated) data in InfluxDB.

### **E. Add Circuit Breaker Pattern (Medium Priority)**

* **Problem:** Repeated API failures aren't handled robustly.
* **Solution:** Implement CircuitBreaker class (as detailed previously) and wrap the core API fetching logic (\_fetch\_ohlcv\_data) within await circuit\_breaker.execute(...).

### **F. Implement Efficient Data Transformation (Medium Priority)**

* **Problem:** Potentially inefficient transformation.
* **Solution:** Optimize the transformation from the Kraken API response format (arrays) to the InfluxDB point format (or dictionaries/DataFrames). Use efficient methods like list comprehensions or optimized loops. Avoid row-by-row DataFrame appends. Parallelization (ProcessPoolExecutor) is likely overkill unless the transformation logic becomes extremely complex and CPU-bound.

### **G. Data Format Support & Versioning (Low/Medium Priority)**

* **Problem:** Only supports Kraken OHLCV; no format versioning.
* **Solution:**
  * **Format Support:** Refactor the fetching and transformation logic to be adapter-based if support for other exchanges or data types (e.g., trades) is anticipated later. The ABC already defines get\_historical\_trades.
  * **Versioning:** Store a format\_version tag/field alongside the data in InfluxDB or in the PostgreSQL metadata table. Implement transformation logic (\_transform\_data\_format) if the storage format needs to change in the future, ensuring backward compatibility during reads.

### **H. Pagination & Large Dataset Handling (Low/Medium Priority)**

* **Problem:** No pagination for API requests; potential memory issues with large datasets.
* **Solution:**
  * **API Pagination:** Kraken's OHLC endpoint uses a since parameter, effectively allowing chronological fetching. Ensure the fetching logic iteratively calls the API using the last timestamp from the previous response until the desired end time is reached, processing data in chunks rather than requesting one massive range.
  * **Memory:** Process and store data in chunks. When querying, allow optional limit parameters and potentially stream results instead of loading everything into memory if datasets become extremely large.

**Conclusion:** The supporting documents confirm the need for a robust historical data pipeline primarily targeting **InfluxDB** for OHLCV storage, as defined in the schema. The solutions proposed in the initial whiteboard remain relevant but should be implemented with InfluxDB as the main persistence target for time-series data. Rate limiting and validation remain critical high-priority items.
