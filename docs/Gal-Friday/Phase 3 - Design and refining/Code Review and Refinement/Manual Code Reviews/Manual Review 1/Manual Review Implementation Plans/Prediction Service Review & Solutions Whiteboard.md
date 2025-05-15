# **Prediction Service (prediction\_service.py): Review Summary & Solution Whiteboard**

## **1\. Summary of Manual Review Findings (prediction\_service.md)**

* **Strengths:** Effective use of ProcessPoolExecutor for non-blocking inference, robust error handling in the pipeline, good validation of expected features, proper async task management (tracking/cancellation), clean event subscription/publishing.
* **Functional Requirements Gaps (High Priority):**
  * **Limited Model Support:** Only supports native XGBoost models. FR-302 requires support for scikit-learn and potentially TensorFlow/PyTorch.
  * **Missing Ensemble Logic:** No implementation for combining predictions from multiple models (FR-307).
  * **Limited Preprocessing:** Only basic type conversion; lacks scaling/normalization steps (FR-304).
  * **No Retraining Integration:** Doesn't implement or integrate with a model retraining pipeline (FR-309-312).
* **Design & Implementation Issues:** Commented-out code remnants, mixed logging approaches (module logger \+ injected service), hard dependency on XGBoost library/methods, lacks model abstraction.
* **Error Handling Concerns:** Inference might proceed with NaN values; model loading failures aren't handled with fallbacks.
* **Configuration Issues:** Limited model configurability beyond path/features, process pool size not configured here, no model reloading mechanism.

## **2\. Whiteboard: Proposed Solutions**

Addressing the high and medium priority recommendations:

### **A. Implement Support for Additional Model Types (High Priority \- FR-302)**

* **Problem:** The service is hardcoded to load and run only native XGBoost models.
* **Solution:**
  1. **Modify Configuration:** Update the configuration (prediction\_service section) to include model\_type (e.g., "xgboost", "sklearn", "tensorflow") alongside model\_path.
  2. **Modify \_run\_inference\_task:** Add a model\_type argument. Inside the function, use if/elif/else based on model\_type to load the model using the appropriate library (xgboost, joblib for scikit-learn, tensorflow.keras.models.load\_model) and call its prediction method (predict, predict\_proba). Ensure the prediction output is standardized (e.g., extracting the relevant probability or value).
  3. **Update Pipeline:** Modify \_run\_prediction\_pipeline to pass the model\_type from the config to loop.run\_in\_executor.

\# In prediction\_service.py

\# \--- Updated Inference Function \---
def \_run\_inference\_task(
    model\_path: str,
    model\_type: str, \# Added
    feature\_vector: np.ndarray,
    model\_feature\_names: List\[str\] \# Still useful for DMatrix or DataFrame creation
) \-\> Dict\[str, Any\]:
    """
    Loads a model based on type and runs inference in a separate process.
    """
    try:
        model\_type \= model\_type.lower()
        input\_data \= feature\_vector.reshape(1, \-1) \# Common reshaping

        if model\_type \== "xgboost":
            import xgboost as xgb \# Import inside function for process safety
            model \= xgb.Booster()
            model.load\_model(model\_path)
            dmatrix \= xgb.DMatrix(input\_data, feature\_names=model\_feature\_names)
            prediction \= model.predict(dmatrix)
            prediction\_float \= float(prediction.item()) \# Assuming single output

        elif model\_type \== "sklearn":
            import joblib \# Import inside function
            model \= joblib.load(model\_path)
            \# Assuming predict\_proba for classifiers, get prob of class 1
            if hasattr(model, "predict\_proba"):
                 prediction \= model.predict\_proba(input\_data)\[0, 1\]
            else: \# Fallback for regressors or other types
                 prediction \= model.predict(input\_data)\[0\]
            prediction\_float \= float(prediction)

        elif model\_type \== "tensorflow":
            \# Ensure TensorFlow is installed in the environment where the executor runs
            import tensorflow as tf \# Import inside function
            model \= tf.keras.models.load\_model(model\_path)
            \# TF predict might return multi-dimensional array
            prediction \= model.predict(input\_data)\[0\]
            \# Assuming the desired output is the first element for classification/regression
            prediction\_float \= float(prediction\[0\])

        else:
            return {"error": f"Unsupported model type: {model\_type}"}

        return {"prediction": prediction\_float}

    except FileNotFoundError:
        return {"error": f"Model file not found: {model\_path}"}
    except ImportError as imp\_err:
         return {"error": f"Missing library for model type '{model\_type}': {imp\_err}"}
    except Exception as e:
        \# Log the full traceback in the worker process if possible,
        \# but return a serializable error message.
        \# import traceback; traceback.print\_exc() \# Optional: print traceback in worker
        return {"error": f"Inference failed for {model\_type} model: {str(e)}"}

\# \--- Update PredictionService Class \---
class PredictionService:
    def \_\_init\_\_(self, config: Dict\[str, Any\], ...):
        \# ...
        self.\_model\_path \= self.\_config.get("model\_path")
        self.\_model\_type \= self.\_config.get("model\_type", "xgboost") \# Get model type from config
        \# ... (check model\_path exists) ...

    async def \_run\_prediction\_pipeline(self, event: FeatureEvent) \-\> None:
        try:
            \# ... (prepare features) ...
            feature\_vector \= self.\_preprocess\_features(event.features) \# Use preprocessing step
            if feature\_vector is None: return

            loop \= asyncio.get\_running\_loop()
            inference\_future: InferenceTaskType \= loop.run\_in\_executor(
                self.\_process\_pool\_executor,
                \_run\_inference\_task,
                self.\_model\_path,
                self.\_model\_type, \# Pass model type
                feature\_vector,
                self.\_model\_feature\_names,
            )
            \# ... (rest of pipeline: track future, await result, publish) ...
        \# ... (exception handling) ...

### **B. Add Feature Preprocessing Logic (High Priority \- FR-304)**

* **Problem:** Features are only converted to float; models often require scaling or normalization applied during training.
* **Solution:**
  1. **Configuration:** Add configuration for preprocessing steps, including the path to saved scaler/transformer objects (e.g., a StandardScaler saved with joblib).
  2. **Implement \_preprocess\_features:** Create this method. It first calls \_prepare\_features\_for\_model to get the ordered numpy array. Then, if configured, it loads the scaler object and applies scaler.transform() to the feature vector. Handle errors during loading or transformation.
  3. **Update Pipeline:** Call \_preprocess\_features instead of \_prepare\_features\_for\_model directly in \_run\_prediction\_pipeline.

\# In PredictionService class

\# Add in \_\_init\_\_
\# self.\_scaler\_path \= self.\_config.get("preprocessing", {}).get("scaler\_path")
\# self.\_scaler \= None \# Lazy load scaler if path exists

def \_load\_scaler(self):
    """Loads the scaler object from the configured path."""
    if self.\_scaler\_path and self.\_scaler is None:
         try:
              import joblib
              self.\_scaler \= joblib.load(self.\_scaler\_path)
              self.logger.info(f"Loaded scaler from {self.\_scaler\_path}", source\_module=self.\_source\_module)
         except FileNotFoundError:
              self.logger.error(f"Scaler file not found: {self.\_scaler\_path}", source\_module=self.\_source\_module)
              self.\_scaler\_path \= None \# Avoid repeated attempts
         except Exception as e:
              self.logger.error(f"Error loading scaler: {e}", source\_module=self.\_source\_module, exc\_info=True)
              self.\_scaler\_path \= None
    return self.\_scaler

def \_preprocess\_features(self, features\_dict: Dict\[str, str\]) \-\> Optional\[np.ndarray\]:
    """Prepares the final feature vector, including scaling if configured."""
    \# 1\. Get ordered numpy array (handles missing features, basic type conversion)
    feature\_vector \= self.\_prepare\_features\_for\_model(features\_dict)
    if feature\_vector is None:
        return None

    \# 2\. Validate quality (e.g., check NaN percentage \- see Section F)
    if not self.\_validate\_feature\_quality(feature\_vector): \# Implement this method
         self.logger.warning("Feature quality validation failed. Skipping prediction.", source\_module=self.\_source\_module)
         return None

    \# 3\. Apply scaling if configured and scaler loaded
    scaler \= self.\_load\_scaler() \# Attempt to load scaler if not already loaded
    if scaler:
        try:
            \# Scaler expects 2D array \[n\_samples, n\_features\]
            feature\_vector\_reshaped \= feature\_vector.reshape(1, \-1)
            \# Apply scaling
            scaled\_vector \= scaler.transform(feature\_vector\_reshaped)
            \# Return the 1D scaled vector
            return scaled\_vector\[0\]
        except ValueError as ve:
             \# Catches shape mismatches or NaN issues during transform
             self.logger.error(f"Error applying scaler transform: {ve}. Check feature count/order and NaN values.", source\_module=self.\_source\_module)
             return None \# Cannot proceed if scaling fails
        except Exception as e:
            self.logger.error(f"Unexpected error during scaling: {e}", source\_module=self.\_source\_module, exc\_info=True)
            return None
    else:
        \# No scaler configured or loaded, return the original (validated) vector
        return feature\_vector

\# Ensure \_validate\_feature\_quality is implemented (see Section F)

### **C. Implement Model Ensemble Logic (High Priority \- FR-307)**

* **Problem:** Only supports a single model; requirements mention combining multiple model outputs.
* **Solution:**
  1. **Configuration:** Define an ensemble section in the config, listing multiple models, each with path, type, feature\_names, and optionally weight.
  2. **Refactor Inference:** Rename \_run\_prediction\_pipeline to \_run\_single\_model\_pipeline which handles one model.
  3. **Implement \_run\_ensemble\_pipeline:** This new method (called by the event handler) iterates through the configured ensemble models. For each model, it calls \_run\_single\_model\_pipeline (or just the inference part) concurrently using asyncio.gather.
  4. **Combine Results:** Collect the successful predictions from asyncio.gather. Apply the configured combination logic (e.g., weighted average based on weights in config).
  5. **Publish:** Publish a single PredictionEvent with the combined result, potentially listing contributing models in metadata.

\# In PredictionService class

async def \_handle\_feature\_event(self, event: FeatureEvent) \-\> None:
    \# ... (checks) ...
    \# Decide whether to run single model or ensemble based on config
    if self.\_config.get("ensemble", {}).get("enabled", False):
         asyncio.create\_task(self.\_run\_ensemble\_pipeline(event))
    elif self.\_model\_path: \# Check if single model is configured
         asyncio.create\_task(self.\_run\_single\_model\_pipeline(event))
    else:
         self.logger.error("No single model path or ensemble configured.", source\_module=self.\_source\_module)

async def \_run\_single\_model\_pipeline(self, event: FeatureEvent, model\_path: Optional\[str\] \= None, model\_type: Optional\[str\] \= None, model\_feature\_names: Optional\[List\[str\]\] \= None) \-\> Optional\[float\]:
     """Runs the prediction pipeline for a single specified model. Returns prediction value or None."""
     \# Use provided params or defaults from self
     m\_path \= model\_path or self.\_model\_path
     m\_type \= model\_type or self.\_model\_type
     m\_features \= model\_feature\_names or self.\_model\_feature\_names

     if not m\_path or not m\_features:
          self.logger.error("Missing model path or features for single model pipeline.", source\_module=self.\_source\_module)
          return None

     try:
          feature\_vector \= self.\_preprocess\_features(event.features) \# Use full preprocessing
          if feature\_vector is None: return None

          loop \= asyncio.get\_running\_loop()
          inference\_future: InferenceTaskType \= loop.run\_in\_executor(
               self.\_process\_pool\_executor, \_run\_inference\_task,
               m\_path, m\_type, feature\_vector, m\_features
          )
          \# Track/await omitted for brevity in this sub-function context, handle in caller
          result \= await inference\_future

          if "error" in result:
               self.logger.error(f"Inference failed for model {m\_path}: {result\['error'\]}", source\_module=self.\_source\_module)
               return None
          elif "prediction" in result:
               return result\["prediction"\]
          else:
               self.logger.error(f"Invalid inference result format from {m\_path}: {result}", source\_module=self.\_source\_module)
               return None
     except Exception as e:
          self.logger.error(f"Error in single model pipeline for {m\_path}: {e}", source\_module=self.\_source\_module, exc\_info=True)
          return None

async def \_run\_ensemble\_pipeline(self, event: FeatureEvent) \-\> None:
    """Runs predictions for multiple models and combines them."""
    ensemble\_config \= self.\_config.get("ensemble", {})
    models\_to\_run \= ensemble\_config.get("models", \[\])
    combination\_method \= ensemble\_config.get("combination\_method", "weighted\_average") \# or 'average', 'voting' etc.

    if not models\_to\_run:
         self.logger.error("Ensemble enabled but no models configured.", source\_module=self.\_source\_module)
         return

    \# Prepare features once (assuming all models use the same base features before specific preprocessing)
    \# Note: If models need different preprocessing, this needs adjustment.
    prepared\_features\_dict \= event.features \# Pass dict, preprocessing happens in \_run\_single\_model\_pipeline if needed there
    \# Or preprocess once if common scaler:
    \# common\_preprocessed\_vector \= self.\_preprocess\_features(event.features)
    \# if common\_preprocessed\_vector is None: return

    inference\_tasks \= \[\]
    model\_configs \= \[\] \# Store config alongside task for weighting later
    for model\_cfg in models\_to\_run:
         path \= model\_cfg.get("path")
         m\_type \= model\_cfg.get("type")
         m\_features \= model\_cfg.get("feature\_names") \# Model-specific features
         weight \= float(model\_cfg.get("weight", 1.0))

         if not all(\[path, m\_type, m\_features\]):
              self.logger.warning(f"Skipping invalid model config in ensemble: {model\_cfg}", source\_module=self.\_source\_module)
              continue

         \# \--- Run inference for each model \---
         \# This re-uses the single model pipeline logic, including preprocessing if needed
         \# If preprocessing differs per model, \_run\_single\_model\_pipeline needs adjustment
         \# or preprocessing needs to happen here based on model\_cfg.
         task \= asyncio.create\_task(
              self.\_run\_single\_model\_pipeline(event, path, m\_type, m\_features)
         )
         inference\_tasks.append(task)
         model\_configs.append({"weight": weight, "id": model\_cfg.get("id", path)})

    \# Gather results
    results \= await asyncio.gather(\*inference\_tasks, return\_exceptions=True)

    \# Combine results
    valid\_predictions \= \[\]
    total\_weight \= Decimal(0)
    contributing\_models \= \[\]
    for i, res in enumerate(results):
         weight \= Decimal(str(model\_configs\[i\]\["weight"\])) \# Use Decimal for weights
         model\_id \= model\_configs\[i\]\["id"\]
         if isinstance(res, float): \# Check if result is the prediction float
              valid\_predictions.append((Decimal(str(res)), weight)) \# Store as Decimal
              total\_weight \+= weight
              contributing\_models.append(model\_id)
         elif isinstance(res, Exception):
              self.logger.error(f"Ensemble model {model\_id} failed: {res}", source\_module=self.\_source\_module)
         \# Else: \_run\_single\_model\_pipeline returned None (already logged error)

    if not valid\_predictions:
         self.logger.error("No valid predictions obtained from ensemble models.", source\_module=self.\_source\_module)
         return

    \# Combine using weighted average
    final\_prediction \= Decimal(0)
    if total\_weight \> 0 and combination\_method \== "weighted\_average":
         final\_prediction \= sum(pred \* w for pred, w in valid\_predictions) / total\_weight
    elif valid\_predictions and combination\_method \== "average": \# Simple average
         final\_prediction \= sum(pred for pred, w in valid\_predictions) / Decimal(len(valid\_predictions))
    \# Add other combination methods like voting if needed

    \# Publish combined prediction
    prediction\_event \= PredictionEvent(
         source\_module=self.\_source\_module,
         event\_id=uuid.uuid4(),
         timestamp=datetime.utcnow(),
         trading\_pair=event.trading\_pair,
         exchange=event.exchange,
         timestamp\_prediction\_for=event.timestamp\_features\_for,
         model\_id=f"ensemble\_{combination\_method}", \# Indicate ensemble
         prediction\_target=self.\_prediction\_target,
         prediction\_value=float(final\_prediction), \# Convert back to float for event
         confidence=None, \# Confidence calculation for ensemble?
         associated\_features={"contributing\_models": contributing\_models} \# Add metadata
    )
    await self.\_publish\_prediction(prediction\_event)

### **D. Add Model Reloading Capability (Medium Priority)**

* **Problem:** No way to load updated models without restarting the service.
* **Solution:** Implement a mechanism (e.g., triggered by a file watcher, a specific event, or a CLI command via CLIService) that signals the PredictionService. Upon receiving the signal, the service could:
  * **Option 1 (Simple):** Update internal state (like \_model\_path). The \_run\_inference\_task function (running in the executor) would then load the model from the updated path *each time* it runs. This avoids complex process management but adds model loading overhead to every prediction.
  * **Option 2 (Executor Restart \- More Complex):** Signal the main application (main.py) to shut down the *current* ProcessPoolExecutor (waiting for existing tasks) and start a *new* one. The PredictionService would then use the new executor, ensuring new processes load the new model. This is cleaner but involves coordinating with main.py.
  * **Option 3 (Shared Memory/Signaling \- Very Complex):** Use inter-process communication or shared memory to signal worker processes to reload their internal model state. Highly complex.
  * **Recommendation:** Start with Option 1 for simplicity, accepting the performance trade-off. Add configuration to enable/disable this dynamic reloading.

\# In PredictionService class (Illustrating Option 1\)

\# Add method to be called externally (e.g., by CLIService or file watcher task)
async def trigger\_model\_reload(self, new\_model\_path: Optional\[str\] \= None) \-\> bool:
    """Updates the model path to be used for subsequent predictions."""
    if new\_model\_path:
         \# Basic check if path exists (optional)
         \# import os; if not os.path.exists(new\_model\_path): ... log error ... return False
         self.\_model\_path \= new\_model\_path
         self.logger.info(f"Updated model path for next inferences: {self.\_model\_path}", source\_module=self.\_source\_module)
         \# If using ensemble, need logic to update specific model path in ensemble config
         return True
    else:
         \# Reload based on existing config path (e.g., if file was overwritten)
         config\_path \= self.\_config.get("model\_path")
         if config\_path and config\_path \!= self.\_model\_path:
              self.\_model\_path \= config\_path
              self.logger.info(f"Model path refreshed from config for next inferences: {self.\_model\_path}", source\_module=self.\_source\_module)
              return True
         elif config\_path:
              self.logger.info("Model path in config is same as current. No reload triggered.", source\_module=self.\_source\_module)
              return False
         else:
              self.logger.error("Cannot reload model: model\_path not found in config.", source\_module=self.\_source\_module)
              return False

\# \_run\_inference\_task remains the same \- it loads the model using the provided path on each call.

### **E. Code Cleanup & Logging Standardization (Medium Priority)**

* **Problem:** Commented-out code, mixed logging approaches.
* **Solution:**
  * Remove all commented-out code blocks related to the old event loop (\_main\_task, etc.).
  * Remove the module-level log \= logging.getLogger(\_\_name\_\_).
  * Ensure *all* logging calls within the class use self.logger (the injected LoggerService instance).

### **F. Improve NaN Handling (Medium Priority)**

* **Problem:** Inference proceeds even if some features are NaN; only checks if *all* are NaN.
* **Solution:** Implement \_validate\_feature\_quality method. This method checks the *percentage* of NaN values in the prepared feature vector. If the percentage exceeds a configurable threshold (prediction\_service.max\_nan\_percentage), return False. Call this validation method within \_preprocess\_features before applying scaling or returning the vector.
  \# In PredictionService class

  def \_validate\_feature\_quality(self, feature\_vector: np.ndarray) \-\> bool:
      """Validates feature vector quality, checking NaN percentage."""
      if feature\_vector is None: return False \# Should not happen if called after \_prepare...

      nan\_count \= np.isnan(feature\_vector).sum()
      if nan\_count \== 0:
          return True \# No NaNs, quality is good

      total\_features \= len(feature\_vector)
      if total\_features \== 0: return False \# Empty vector

      nan\_pct \= (nan\_count / total\_features) \* 100
      max\_nan\_pct \= self.\_config.get("max\_nan\_percentage", 20.0) \# Configurable threshold

      if nan\_pct \> max\_nan\_pct:
          self.logger.warning(
              f"Feature quality issue: {nan\_pct:.1f}% NaN values detected (threshold: {max\_nan\_pct:.1f}%). Skipping prediction.",
              source\_module=self.\_source\_module
          )
          return False
      else:
          self.logger.debug(
              f"NaN features present ({nan\_pct:.1f}%) but below threshold ({max\_nan\_pct:.1f}%). Proceeding with prediction.",
              source\_module=self.\_source\_module
          )
          \# Optionally, consider imputation strategies here if needed, instead of just proceeding
          return True

  \# Ensure \_preprocess\_features calls this:
  \# def \_preprocess\_features(...) \-\> Optional\[np.ndarray\]:
  \#     feature\_vector \= self.\_prepare\_features\_for\_model(...)
  \#     if feature\_vector is None: return None
  \#
  \#     if not self.\_validate\_feature\_quality(feature\_vector): \# \<--- CALL VALIDATION HERE
  \#          return None
  \#
  \#     \# ... rest of preprocessing (scaling) ...

**Conclusion:** Implementing support for multiple model types and robust feature preprocessing are the highest priorities. Adding ensemble logic and improving NaN handling will further enhance functionality and reliability. Cleaning up the code and standardizing logging are important for maintainability.
