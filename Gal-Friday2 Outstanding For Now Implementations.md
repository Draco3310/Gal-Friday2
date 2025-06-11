# **Gal-Friday2: Outstanding "For Now" Implementations**

This document tracks temporary or placeholder implementations within the Gal-Friday2 codebase, marked with "for now" comments. The goal is to provide a structured overview of these areas, their current limitations, and proposed next steps for full implementation. This will serve as a targeted list for future development and refactoring, potentially leveraging AI tools like Codex for assistance.

## **I. Core Module**

### **gal\_friday/core/asset\_registry.py**

* **Line:** 277,44  
* **Current Comment:** \# Future expansion examples (commented for now)  
* **Problem/Context:** The asset registry has commented-out sections indicating planned future expansion for handling different asset types or more complex registration logic. These are currently unimplemented placeholders.  
* **Proposed Next Steps:** \* **Identify Specific Needs:** Define concrete requirements for future asset types or registry functionalities.  
  * **Implement Fully:** Develop and integrate the commented-out or new features based on the identified needs.  
  * **Remove Comment:** Once implemented, remove the "for now" comment.

### **gal\_friday/core/event\_store.py**

* **Line:** 109,23  
* **Current Comment:** \# For now, let's attempt to get a default or raise  
* **Problem/Context:** This indicates a temporary error handling or default value retrieval strategy when an event is not found or cannot be processed as expected. It's a simplified approach that might not be robust.  
* **Proposed Next Steps:** \* **Refine Error Handling:** Implement a more specific and robust error handling mechanism (e.g., custom exceptions for different types of event store failures).  
  * **Define Default Behavior:** Clearly define and implement the intended default behavior if an event is not found, rather than a generic "attempt to get a default or raise."

## **II. Data Access Layer (DAL) & Migrations**

### **gal\_friday/dal/migrations/migration\_manager.py**

* **Line:** 39,15  
* **Current Comment:** \# For now, subsequent calls will likely fail if alembic\_cfg cannot be loaded.  
* **Problem/Context:** The current handling of alembic\_cfg loading is fragile. If the configuration cannot be loaded, subsequent migration operations might fail without proper error recovery.  
* **Proposed Next Steps:** \* **Robust Configuration Loading:** Implement comprehensive error handling and fallback mechanisms for alembic\_cfg loading. Ensure the system can gracefully handle missing or malformed configuration files.  
  * **Early Validation:** Add validation checks at the beginning of migration operations to ensure alembic\_cfg is successfully loaded before proceeding.  
* **Line:** 103,15  
* **Current Comment:** \# For now, let's try to get script heads which doesn't require DB connection.  
* **Problem/Context:** This implies a workaround to get Alembic script heads without a database connection, which might be necessary for certain CLI commands but could lead to inconsistencies if the script heads in the environment don't match the database state.  
* **Proposed Next Steps:** \* **Conditional DB Connection:** Determine scenarios where a DB connection is strictly necessary for migration operations and enforce it.  
  * **Clear Separation:** Explicitly separate "offline" migration operations (script analysis) from "online" operations (applying migrations).  
* **Line:** 112,15  
* **Current Comment:** \# For now, sticking to a method that can get DB state if possible.  
* **Problem/Context:** Similar to the above, this suggests a less-than-ideal method for retrieving the database state. It might be unreliable or inefficient.  
* **Proposed Next Steps:** \* **Standardize DB State Retrieval:** Implement a robust and consistent method for retrieving the database's current migration state, ensuring it's always accurate and handles various database connection scenarios.

### **gal\_friday/dal/models/position\_adjustment.py**

* **Line:** 23,167  
* **Current Comment:** reconciliation\_id: Mapped\[UUID | None\] \= mapped\_column( \# Schema in 001 allows NULL, 003 implies NOT NULL via REFERENCES. Assuming 001 version for FK nullability for now.  
* **Problem/Context:** There's an inconsistency between schema versions regarding the nullability of reconciliation\_id. The current code assumes an older schema (001) for foreign key nullability, which might break if a newer schema (003) with NOT NULL is applied.  
* **Proposed Next Steps:** \* **Align Schema Definitions:** Resolve the schema inconsistency. Either update the database schema to explicitly allow NULL or enforce NOT NULL and adjust the code to handle the presence of a reconciliation\_id consistently.  
  * **Add Migration:** If changing the schema, create a new Alembic migration to reflect the correct nullability.

### **gal\_friday/dal/repositories/experiment\_repository.py**

* **Line:** 67,15  
* **Current Comment:** \# For now, let's assume BaseRepository.create or a direct session.merge could work  
* **Problem/Context:** This indicates uncertainty or a lack of clarity in how new experiment records are persisted. Relying on assumptions can lead to subtle data integrity issues.  
* **Proposed Next Steps:** \* **Explicit Persistence Logic:** Clearly define and implement the exact SQLAlchemy operation (session.add, session.merge, etc.) for creating new Experiment records, considering all edge cases (e.g., existing records, detached objects).  
  * **Unit Tests:** Add specific unit tests to ensure experiment creation and updates behave as expected with the chosen persistence method.  
* **Line:** 134,15  
* **Current Comment:** \# For now, let's just return the input or fetch it.  
* **Problem/Context:** This suggests a simplified or incomplete implementation for retrieving experiment outcomes. It might not accurately reflect the stored data or handle various query parameters.  
* **Proposed Next Steps:** \* **Implement Robust Retrieval:** Develop proper query logic to fetch experiment outcomes from the database, handling different filtering, sorting, and pagination requirements.  
  * **Error Handling:** Ensure the method gracefully handles cases where no matching experiment outcomes are found.

### **gal\_friday/dal/alembic.ini**

* **Line:** 25,3  
* **Current Comment:** \# For now, "." is standard if CWD is project root.  
* **Problem/Context:** The alembic.ini file's path resolution (.) assumes the current working directory is always the project root, which might not be true in all deployment or development environments (e.g., when running scripts from subdirectories).  
* **Proposed Next Steps:** \* **Relative Path Configuration:** Use more robust path resolution in alembic.ini or ensure that scripts running Alembic explicitly set the working directory or provide the correct path to alembic.ini.  
  * **Environment Variables:** Consider using environment variables to specify the project root or Alembic configuration path.

## **III. Execution Module**

### **gal\_friday/execution/adapters.py**

* **Line:** 543,11  
* **Current Comment:** \# For now, place individually but could be enhanced with AddOrderBatch  
* **Problem/Context:** Orders are currently placed individually, which might be inefficient for high-frequency trading where batching orders could significantly reduce latency and API calls.  
* **Proposed Next Steps:** \* **Implement Batching:** Integrate with the exchange's AddOrderBatch functionality (if available) or implement a local batching mechanism to aggregate orders before sending them to the exchange API.  
  * **Performance Testing:** Measure the performance improvements after implementing batching.

### **gal\_friday/execution/order\_position\_integration.py**

* **Line:** 417,15  
* **Current Comment:** \# For now, just log that manual intervention is needed  
* **Problem/Context:** This indicates a critical state where an automated process fails and requires manual intervention. While logging is good, it's not a complete solution for a production system.  
* **Proposed Next Steps:** \* **Automated Recovery/Alerting:** Implement automated recovery strategies or robust alerting (e.g., email, PagerDuty) to notify operations teams immediately, providing all necessary context for manual intervention.  
  * **Define Manual Process:** Document the exact manual intervention steps required.  
* **Line:** 437,15  
* **Current Comment:** \# For now, placeholder implementation  
* **Problem/Context:** This suggests a function or method crucial for order-position integration is not fully implemented and currently has only basic or non-functional logic.  
* **Proposed Next Steps:** \* **Full Implementation:** Develop the complete logic for this placeholder, ensuring it adheres to business rules and integrates correctly with other components.  
  * **Comprehensive Testing:** Write unit and integration tests to validate the full implementation.

### **gal\_friday/execution/websocket\_client.py**

* **Line:** 709,15  
* **Current Comment:** \# For now, publish as execution report  
* **Problem/Context:** This implies that certain messages received via the WebSocket are currently generically published as "execution reports," even if they might represent different types of events (e.g., order updates, fill confirmations, cancellations). This could lead to a loss of granularity or incorrect event handling downstream.  
* **Proposed Next Steps:** \* **Granular Event Mapping:** Accurately parse WebSocket messages and map them to specific internal event types (e.g., OrderUpdateEvent, FillEvent, CancellationEvent).  
  * **Dedicated Event Publishing:** Publish these granular events to the appropriate PubSub topics, allowing downstream services to react precisely to each event type.

## **IV. Model Lifecycle & Prediction**

### **gal\_friday/model\_lifecycle/experiment\_manager.py**

* **Line:** 1019,81  
* **Current Comment:** \# Add to active experiments (still using ExperimentConfig dataclass for now)  
* **Problem/Context:** The ExperimentConfig dataclass might be a simplified representation and may need to evolve into a more comprehensive model as the experiment management system matures.  
* **Proposed Next Steps:** \* **Evaluate ExperimentConfig:** Review ExperimentConfig to determine if it meets all long-term requirements. If not, design a more robust model.  
  * **Migration:** If a new model is introduced, plan a migration strategy for existing data.  
* **Line:** 1405,19  
* **Current Comment:** \# For now, assume a basic mapping or that ExperimentConfig adapts.  
* **Problem/Context:** This implies a loose coupling or a simplified mapping between data structures related to experiments. This could lead to data inconsistencies or errors if the assumptions are violated.  
* **Proposed Next Steps:** \* **Explicit Data Mapping:** Implement explicit and robust data mapping logic between different experiment-related data structures.  
  * **Validation:** Add validation to ensure data consistency during mapping.

## **V. Models & Configuration**

### **gal\_friday/models/configuration.py**

* **Line:** 54,26  
* **Current Comment:** \# Returning dict for now to satisfy type hint via forward reference  
* **Problem/Context:** A dictionary is being returned to satisfy type hints, likely as a temporary solution. This might bypass proper object serialization/deserialization or obscure the actual data structure.  
* **Proposed Next Steps:** \* **Proper Serialization:** Implement proper serialization/deserialization methods (e.g., to\_dict, from\_dict, Pydantic models) that return the actual intended object types rather than raw dictionaries.  
  * **Refactor Type Hints:** Update type hints to reflect the actual object types.

### **gal\_friday/models/order.py**

* **Line:** 103,26  
* **Current Comment:** \# Returning dict for now  
* **Problem/Context:** Similar to configuration.py, returning a dictionary for an Order object is a temporary measure that might hide structured data.  
* **Proposed Next Steps:** \* **Implement to\_dict or to\_json:** Provide a structured method to convert Order objects into a dictionary format if needed for specific use cases (e.g., API responses, logging), ensuring all relevant fields are included and properly typed.

### **gal\_friday/models/signal.py**

* **Line:** 78,26  
* **Current Comment:** \# Returning dict for now  
* **Problem/Context:** Similar to other models, returning a dictionary for a Signal object suggests an incomplete or simplified data representation.  
* **Proposed Next Steps:** \* **Implement to\_dict or to\_json:** Ensure Signal objects have a proper method for structured serialization to a dictionary, if required, maintaining data integrity and clarity.

### **gal\_friday/models/trade.py**

* **Line:** 99,26  
* **Current Comment:** \# Returning dict for now  
* **Problem/Context:** Similar to other models, returning a dictionary for a Trade object suggests an incomplete or simplified data representation.  
* **Proposed Next Steps:** \* **Implement to\_dict or to\_json:** Ensure Trade objects have a proper method for structured serialization to a dictionary, if required, maintaining data integrity and clarity.

## **VI. Monitoring & Dashboard**

### **gal\_friday/monitoring/dashboard\_backend.py**

* **Line:** 244,11  
* **Current Comment:** \# For now, calculate based on position concentration  
* **Problem/Context:** The current calculation of a metric (likely related to risk or portfolio health) is simplified and only considers position concentration. A more comprehensive calculation might be needed.  
* **Proposed Next Steps:** \* **Refine Calculation Logic:** Expand the calculation to include other relevant factors (e.g., volatility, market liquidity, correlations) to provide a more accurate and holistic view.  
  * **External Library Integration:** Consider integrating a specialized financial analytics library for advanced risk metrics.  
* **Line:** 441,7  
* **Current Comment:** \# For now, return mock data  
* **Problem/Context:** The dashboard backend is currently returning mock data, meaning it's not yet connected to real-time or historical data sources.  
* **Proposed Next Steps:** \* **Integrate with Data Sources:** Connect the backend to the actual data repositories (e.g., database, InfluxDB) to fetch live or historical performance metrics.  
  * **Remove Mock Data:** Delete the mock data generation once real data integration is complete.

### **gal\_friday/monitoring/position\_order\_data\_quality.py**

* **Line:** 347,15  
* **Current Comment:** \# For now, placeholder for future enhancement  
* **Problem/Context:** A specific data quality check or enhancement related to position and order data is currently a placeholder, indicating a missing crucial validation or processing step.  
* **Proposed Next Steps:** \* **Define Data Quality Rules:** Clearly define the data quality rules and metrics for position and order data (e.g., consistency, completeness, timeliness).  
  * **Implement Validation Logic:** Develop the full implementation of the data quality checks and integrate them into the monitoring pipeline.

### **gal\_friday/monitoring\_service.py**

* **Line:** 1287,15  
* **Current Comment:** \# For now, just log the notification  
* **Problem/Context:** Alerts or important notifications are only being logged, which is insufficient for critical events in a trading system.  
* **Proposed Next Steps:** \* **Implement Robust Alerting:** Integrate with external alerting systems (e.g., PagerDuty, Slack, email) for critical notifications, ensuring high-priority alerts are delivered reliably.  
  * **Configurable Alerting:** Allow configuration of alert thresholds and delivery channels.  
* **Line:** 2737,15  
* **Current Comment:** \# For now, assuming it's sync as requested for MVP.  
* **Problem/Context:** A process or operation is currently implemented synchronously, which might block the main thread and hinder performance, especially in a high-frequency system. This was likely a decision for MVP.  
* **Proposed Next Steps:** \* **Asynchronous Implementation:** Refactor the operation to be asynchronous (e.g., using asyncio, Celery, or a dedicated message queue) to prevent blocking and improve scalability.  
  * **Performance Benchmarking:** Benchmark synchronous vs. asynchronous performance to validate the improvement.  
* **Line:** 3139,15  
* **Current Comment:** \# For now, simulating the call  
* **Problem/Context:** A critical function call (possibly to an external service or a complex internal component) is currently simulated, meaning its actual integration is pending.  
* **Proposed Next Steps:** \* **Replace Simulation with Actual Call:** Integrate the real external service or internal component, replacing the simulated call.  
  * **Integration Tests:** Develop robust integration tests for this component.  
* **Line:** 3356,15  
* **Current Comment:** \# For now, we don't have a direct storage mechanism for this  
* **Problem/Context:** A specific type of data (unspecified in the snippet, but likely related to monitoring metrics or alerts) is not being persisted due to a lack of a storage mechanism.  
* **Proposed Next Steps:** \* **Identify Storage Needs:** Determine the appropriate storage solution (e.g., database, InfluxDB, file system) for this specific data.  
  * **Implement Persistence:** Develop the necessary DAL components and logic to store this data reliably.

## **VII. Portfolio Management**

### **gal\_friday/portfolio/position\_manager.py**

* **Line:** 125,11  
* **Current Comment:** \# For now, let's assume it just logs the number of active positions.  
* **Problem/Context:** The current implementation for managing active positions is limited to just logging, rather than actively managing or verifying the state of positions.  
* **Proposed Next Steps:** \* **Active Position Management:** Implement full logic to actively track, update, and reconcile active positions with the exchange.  
  * **State Machine:** Consider a state machine for position lifecycle management (e.g., Open, Closed, Pending Adjustment).  
* **Line:** 392,20  
* **Current Comment:** \# For now, assume we create a new one if the existing is inactive.  
* **Problem/Context:** This implies a simplified logic for handling position creation or activation, potentially overlooking complex scenarios or requiring explicit checks.  
* **Proposed Next Steps:** \* **Refine Position Logic:** Clearly define the conditions for creating a new position versus reactivating or adjusting an existing one. Avoid implicit assumptions.  
  * **Error Handling:** Add checks for unexpected states (e.g., attempting to activate an already active position).  
* **Line:** 407,11  
* **Current Comment:** \# For now, ID will be auto-generated by DB if not provided.  
* **Problem/Context:** While common, relying solely on auto-generated IDs for positions might lead to issues in distributed systems or during backtesting if explicit ID management is required.  
* **Proposed Next Steps:** \* **Strategy for ID Generation:** Decide if a more explicit ID generation strategy is needed (e.g., UUIDs generated by the application) to ensure uniqueness and traceability across services, even before database persistence.  
* **Line:** 433,30  
* **Current Comment:** raise \# Re-raise for now  
* **Problem/Context:** A generic raise without specific error handling means that exceptions are not being caught, processed, or logged properly at this level.  
* **Proposed Next Steps:** \* **Specific Exception Handling:** Catch the appropriate exceptions, log them with context, and re-raise a more specific or custom exception if necessary for higher-level handling.  
  * **Error Recovery:** Implement a recovery or retry mechanism if the error is transient.  
* **Line:** 439,54  
* **Current Comment:** \# Or, iterate all positions if not too many. For now, simplified:  
* **Problem/Context:** This indicates a simplified approach to iterating positions, possibly suggesting that for a large number of positions, this method would be inefficient.  
* **Proposed Next Steps:** \* **Optimized Position Iteration:** Implement a more efficient method for iterating or querying positions, especially for large portfolios (e.g., pagination, indexed queries).  
  * **Performance Considerations:** Analyze the performance impact of this iteration as the number of positions grows.  
* **Line:** 451,7  
* **Current Comment:** \# For now, keeping if the internal logic still uses them to construct TradeInfo objects if needed.  
* **Problem/Context:** This suggests redundant data or objects are being kept "just in case," indicating potential code smells or unclear responsibilities.  
* **Proposed Next Steps:** \* **Code Cleanup:** Refactor the internal logic to ensure that TradeInfo objects are constructed only when and where needed, eliminating redundant data storage or processing.  
  * **Clear Dependencies:** Explicitly define which components require TradeInfo and how they obtain it.

### **gal\_friday/portfolio/reconciliation\_service.py**

* **Line:** 1219,9  
* **Current Comment:** For now, this method prepares the 'correction' dict for the report.  
* **Problem/Context:** The method is currently limited to just preparing a dictionary for a report, implying that the actual reconciliation or correction application is missing or handled elsewhere.  
* **Proposed Next Steps:** \* **Full Reconciliation Logic:** Implement the complete reconciliation logic, including applying corrections, updating database records, and generating audit trails.  
  * **Clear Responsibility:** Ensure this method's responsibility is clearly defined, either solely for reporting or for applying changes.  
* **Line:** 1295,15  
* **Current Comment:** \# For now, only saving auto\_corrections as explicit adjustments.  
* **Problem/Context:** Only automatically determined corrections are being saved as adjustments, potentially ignoring manual corrections or other types of reconciliation actions.  
* **Proposed Next Steps:** \* **Comprehensive Adjustment Handling:** Extend the saving mechanism to handle all types of reconciliation adjustments (automatic, manual, system-generated) consistently.  
  * **Audit Trail:** Ensure a comprehensive audit trail is maintained for all adjustments.  
* **Line:** 1387,15  
* **Current Comment:** \# For now, let's assume it can fetch the model or we adapt.  
* **Problem/Context:** This implies a weak link in fetching or adapting a model within the reconciliation process, relying on an assumption that might not hold true.  
* **Proposed Next Steps:** \* **Robust Model Retrieval:** Implement explicit and robust logic for fetching the correct model (e.g., from a model registry) based on reconciliation context.  
  * **Adaptation Strategy:** Clearly define how the system adapts if the expected model is unavailable or in an incompatible format.

## **VIII. Prediction Service**

### **gal\_friday/prediction\_service.py**

* **Line:** 1376,19  
* **Current Comment:** \# For now, we pass it as np.nan.  
* **Problem/Context:** Missing or unavailable data is currently being passed as np.nan (Not a Number) to the prediction model. While common, this might mask underlying data quality issues or lead to suboptimal predictions if not handled specifically by the model.  
* **Proposed Next Steps:** \* **Robust Imputation:** Implement a more sophisticated data imputation strategy for missing values within the prediction pipeline, potentially leveraging the feature\_imputation module.  
  * **Model Compatibility:** Verify that the prediction models are robust to np.nan or that imputation occurs upstream.

## **IX. Market Price Service**

### **gal\_friday/simulated\_market\_price\_service.py**

* **Line:** 629,30  
* **Current Comment:** \# Fallback to linear for now  
* **Problem/Context:** The system uses a simplified linear interpolation as a fallback, possibly for price data. This might not be accurate or robust enough for all scenarios.  
* **Proposed Next Steps:** \* **Review Interpolation Strategy:** Evaluate if linear interpolation is appropriate for all use cases. Consider more advanced interpolation methods if higher accuracy is needed (e.g., cubic spline, nearest neighbor).  
  * **Define Fallback Conditions:** Clearly define when and why this fallback is used and under what conditions it might be insufficient.  
* **Line:** 635,30  
* **Current Comment:** \# Fallback to linear for now  
* **Problem/Context:** Another instance of linear interpolation fallback, suggesting a systemic issue or a pervasive simplified approach.  
* **Proposed Next Steps:** (Same as above)

## **X. Strategy Module**

### **gal\_friday/strategy\_arbitrator.py**

* **Line:** 2128,15  
* **Current Comment:** \# For now, just log the outcome  
* **Problem/Context:** The outcome of a strategy arbitration decision is only being logged. For a critical component like a strategy arbitrator, more proactive handling is typically required.  
* **Proposed Next Steps:** \* **Proactive Outcome Handling:** Implement actions beyond just logging, such as publishing a specific event (e.g., StrategySelectionEvent), triggering an alert, or updating a dashboard metric.  
  * **Auditability:** Ensure that all arbitration outcomes are easily auditable.

### **gal\_friday/strategy\_selection.py**

* **Line:** 493,23  
* **Current Comment:** \# Placeholder for now \- should integrate with DAL  
* **Problem/Context:** This indicates that strategy selection logic is currently using a placeholder for data that should be retrieved from the Data Access Layer (DAL).  
* **Proposed Next Steps:** \* **DAL Integration:** Implement the necessary DAL calls to fetch actual strategy-related data (e.g., performance metrics, configuration) from the database.  
  * **Remove Placeholder:** Replace the placeholder logic with actual data retrieval.  
* **Line:** 502,23  
* **Current Comment:** \# Placeholder for now \- should integrate with DAL  
* **Problem/Context:** Another instance of placeholder logic requiring DAL integration.  
* **Proposed Next Steps:** (Same as 493,23)  
* **Line:** 512,23  
* **Current Comment:** \# Placeholder for now \- should integrate with DAL  
* **Problem/Context:** Another instance of placeholder logic requiring DAL integration.  
* **Proposed Next Steps:** (Same as 493,23)

## **XI. Utilities & Miscellaneous**

### **gal\_friday/utils/init.py**

* **Line:** 34,7  
* **Current Comment:** \# For now, let the handlers manage the default if it remains None.  
* **Problem/Context:** Default value handling is being deferred to individual handlers, which might lead to inconsistent or implicit behavior across the system.  
* **Proposed Next Steps:** \* **Centralized Default Handling:** Define and apply default values for configurations or parameters at a more centralized point, ensuring consistent behavior.  
  * **Explicit Defaults:** Make default values explicit in function signatures or configuration schemas.

## **XII. Training Script**

### **scripts/train\_initial\_model.py**

* **Line:** 427,57  
* **Current Comment:** \# this function. Simplified: fit only on train data for now  
* **Problem/Context:** The model fitting process is simplified to only use training data, potentially overlooking validation or testing data during initial fitting, which is crucial for preventing overfitting and evaluating generalization.  
* **Proposed Next Steps:** \* **Train/Validation/Test Split:** Ensure a proper train, validation, and test split is applied during model fitting.  
  * **Cross-Validation:** Implement cross-validation techniques for more robust model evaluation during training.  
  * **Hyperparameter Tuning:** Incorporate hyperparameter tuning with validation data.

## **Removed Entries (Compared to previous list)**

The following entries were present in the previous for\_now.md but are no longer in the list you provided, suggesting they might have been addressed or are no longer considered "for now" issues:

* E:CodingGal-Friday2gal\_fridayfeature\_engine.py (all entries)  
  * Line 2253,15: \# For now, create a structured file-based persistence  
  * Line 4154,15: \# For now, simulate historical analysis with OHLCV patterns  
  * Line 4206,15: \# For now, use simplified cross-feature relationships  
  * Line 4285,63: """Impute using machine learning patterns (simplified for now)."""  
  * Line 4288,15: \# For now, implement a simple pattern-based approach  
* E:CodingGal-Friday2gal\_fridaymonitoring\_service.py  
  * Line 3201,15: \# For now, we'll simulate GARCH calculation as it requires additional dependencies  
  * Line 3244,11: \# For now, return None to indicate no data available  
* E:CodingGal-Friday2gal\_fridayportfolio\_manager.py  
  * Line 351,77: self.\_handle\_order\_cancellation(event) \# Still handled locally for now  
* E:CodingGal-Friday2\\gal\_friday\\main.py  
  * Line 1212,71: self.services: list\[Any\] \= \[\] \# UP006: List \-\> list; Use Any for now, can refine later  
* E:Coding\\Gal-Friday2\\gal\_friday\\simulated\_market\_price\_service.py  
  * Line 214,11: \# For now, we'll use mock providers that integrate with existing data  
  * Line 279,11: \# For now, return empty list \- would integrate with actual providers  
  * Line 880,11: \# For now, return empty list  
* E:Coding\\Gal-Friday2\\gal\_friday\\strategy\_arbitrator.py  
  * Line 2062,15: \# For now, just log the outcome