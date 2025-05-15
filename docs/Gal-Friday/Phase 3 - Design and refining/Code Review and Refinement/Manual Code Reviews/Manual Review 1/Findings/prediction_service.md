# Manual Code Review Findings: `prediction_service.py`

## Review Date: May 5, 2025
## Reviewer: AI Assistant
## File Reviewed: `src/gal_friday/prediction_service.py`

## Summary

The `prediction_service.py` module implements the machine learning inference component of the trading system. It successfully handles the core functionality of consuming feature events, running model inference using XGBoost, and publishing prediction events. The implementation offloads CPU-intensive inference tasks to a separate process pool to avoid blocking the main event loop, which is a good practice.

While the module has several strengths, including robust error handling and proper management of asynchronous operations, there are areas that need improvement. The implementation is currently limited to XGBoost models only, missing support for other model types mentioned in the requirements. Additionally, the module lacks some configurability options and doesn't include model ensemble functionality or retraining capabilities.

## Strengths

1. **Proper Process Isolation**: Effectively uses ProcessPoolExecutor to offload CPU-intensive inference tasks, preventing event loop blocking.

2. **Robust Error Handling**: Comprehensive error handling throughout the prediction pipeline, with appropriate logging of issues.

3. **Feature Validation**: Good validation of incoming feature vectors against expected model features, with appropriate handling of missing or invalid features.

4. **Task Management**: Proper tracking and cancellation of inference tasks during shutdown.

5. **Clean Event Interface**: Well-implemented event subscription and publishing mechanism.

## Issues Identified

### A. Functional Requirements Gaps

1. **Limited Model Support**: The implementation only supports XGBoost models, but FR-302 specifies support for scikit-learn and potentially TensorFlow/PyTorch for LSTM models.

2. **Missing Model Ensemble Logic**: No implementation of model ensemble logic (weighted averaging, voting) as mentioned in FR-307.

3. **Limited Feature Preprocessing**: Only converts feature values to float without additional preprocessing like scaling or normalization as potentially required by FR-304.

4. **No Model Retraining Pipeline**: The module doesn't implement the model retraining pipeline specified in FR-309 through FR-312.

### B. Design & Implementation Issues

1. **Commented-Out Code**: Several sections contain commented-out code related to the main task and event loop, suggesting incomplete refactoring:
   ```python
   # self._main_task = asyncio.create_task(self._run_event_loop()) # Remove loop
   ```

2. **Mixing of Logging Approaches**: Uses both a module-level logger (`log`) and the injected logger service (`self.logger`), which could lead to inconsistent logging patterns.

3. **Hard Requirement on XGBoost**: Direct imports and direct use of XGBoost-specific functionality makes it difficult to support other model types.

4. **Lack of Model Interface Abstraction**: No interface to abstract away model-specific details, making it harder to extend to other model types.

### C. Error Handling Concerns

1. **NaN Handling**: While the code checks for and handles missing features by inserting NaN values, inference might still proceed with some NaN values, which could lead to unreliable predictions.

2. **Simplistic All-NaN Check**: The function only checks if all values are NaN, but even a few NaN values might significantly impact prediction quality.

3. **Limited Recovery Options**: If model loading fails, there's no fallback mechanism or retry logic.

### D. Configuration & Hardcoding Issues

1. **Limited Model Configurability**: While model path and feature names are configurable, other model parameters like prediction thresholds are not.

2. **No Configuration for Process Pool Size**: The process pool executor is passed in, but there's no explicit configuration for the optimal pool size based on the system resources.

3. **No Configuration for Model Reloading**: No mechanism to reload models without restarting the service.

## Recommendations

### High Priority

1. **Implement Support for Additional Model Types**:
   ```python
   def _run_inference_task(
       model_path: str,
       model_type: str,  # Add model type parameter
       feature_vector: np.ndarray,
       model_feature_names: List[str]
   ) -> Dict[str, Any]:
       """Runs inference based on model type."""
       try:
           if model_type == "xgboost":
               # Existing XGBoost code
               model = xgb.Booster()
               model.load_model(model_path)
               # ...
           elif model_type == "sklearn":
               import joblib
               model = joblib.load(model_path)
               prediction = model.predict_proba(feature_vector.reshape(1, -1))[0, 1]
               return {"prediction": float(prediction)}
           elif model_type == "tensorflow":
               import tensorflow as tf
               model = tf.keras.models.load_model(model_path)
               prediction = model.predict(feature_vector.reshape(1, -1))[0, 0]
               return {"prediction": float(prediction)}
           else:
               return {"error": f"Unsupported model type: {model_type}"}
       except Exception as e:
           return {"error": f"Inference failed: {str(e)}"}
   ```

2. **Add Feature Preprocessing Logic**:
   ```python
   def _preprocess_features(self, features_dict: Dict[str, str]) -> Optional[np.ndarray]:
       """Preprocess features according to model requirements."""
       # First convert to the right order and data type
       feature_vector = self._prepare_features_for_model(features_dict)
       if feature_vector is None:
           return None

       # Apply model-specific preprocessing
       preprocessing_config = self._config.get("preprocessing", {})

       # Apply scaling if configured
       if preprocessing_config.get("apply_scaling", False):
           scaler_path = preprocessing_config.get("scaler_path")
           if scaler_path:
               try:
                   import joblib
                   scaler = joblib.load(scaler_path)
                   feature_vector = scaler.transform(feature_vector.reshape(1, -1))[0]
               except Exception as e:
                   self.logger.error(f"Error applying scaling: {e}", source_module=self._source_module)

       return feature_vector
   ```

3. **Implement Model Ensemble Logic**:
   ```python
   async def _run_ensemble_prediction(self, feature_vector: np.ndarray) -> Dict[str, float]:
       """Run prediction across multiple models and combine results."""
       models = self._config.get("ensemble", {}).get("models", [])
       if not models:
           # Fall back to single model if no ensemble configured
           return await self._run_single_model_prediction(
               self._model_path,
               self._model_id,
               feature_vector
           )

       # Run predictions for all models in parallel
       prediction_futures = []
       for model_config in models:
           model_path = model_config.get("path")
           model_id = model_config.get("id", "unknown")
           model_type = model_config.get("type", "xgboost")
           model_weight = float(model_config.get("weight", 1.0))

           future = self._run_single_model_prediction(model_path, model_id, model_type, feature_vector)
           prediction_futures.append((future, model_weight))

       # Gather results
       results = []
       total_weight = 0
       for future_tuple in prediction_futures:
           future, weight = future_tuple
           result = await future
           if "prediction" in result:
               results.append((result["prediction"], weight))
               total_weight += weight

       # Apply weighted average
       if total_weight > 0:
           ensemble_prediction = sum(pred * weight for pred, weight in results) / total_weight
           return {"prediction": ensemble_prediction}
       else:
           return {"error": "All models in ensemble failed"}
   ```

### Medium Priority

1. **Add Model Reloading Capability**:
   ```python
   async def reload_model(self) -> bool:
       """Reload the current model from disk (e.g., after retraining)."""
       try:
           # Simply update the model path to trigger reload on next inference
           model_path = self._config.get("model_path")
           if not model_path:
               self.logger.error("Cannot reload model: no model_path configured")
               return False

           self._model_path = model_path
           self.logger.info(f"Model will be reloaded from {model_path} on next inference")
           return True
       except Exception as e:
           self.logger.error(f"Error preparing for model reload: {e}",
                          source_module=self._source_module)
           return False
   ```

2. **Clean Up Commented-Out Code and Standardize Logging**:
   ```python
   # Remove commented-out code:
   # if self._main_task:
   #    ...
   # self._main_task = None

   # Standardize on logger_service:
   # Remove module-level logger:
   # log = logging.getLogger(__name__)
   ```

3. **Improve NaN Handling**:
   ```python
   def _validate_feature_quality(self, feature_vector: np.ndarray) -> bool:
       """Validate feature quality beyond just checking for all NaNs."""
       # Check for any NaN values
       nan_count = np.isnan(feature_vector).sum()
       if nan_count > 0:
           # Calculate what percentage of features are NaN
           nan_pct = nan_count / len(feature_vector) * 100

           # Get configuration threshold (default 20%)
           max_nan_pct = self._config.get("max_nan_percentage", 20.0)

           if nan_pct > max_nan_pct:
               self.logger.warning(
                   f"Too many NaN features: {nan_pct:.1f}% (threshold: {max_nan_pct}%)",
                   source_module=self._source_module
               )
               return False
           else:
               self.logger.debug(
                   f"Some NaN features ({nan_pct:.1f}%) but below threshold of {max_nan_pct}%",
                   source_module=self._source_module
               )
       return True
   ```

### Low Priority

1. **Add Model Performance Monitoring**:
   ```python
   def _track_prediction_performance(self, prediction: float, features: Dict[str, str]) -> None:
       """Track prediction metrics for monitoring."""
       if not hasattr(self, "_prediction_stats"):
           self._prediction_stats = {
               "count": 0,
               "sum": 0,
               "sum_squared": 0,
               "min": float('inf'),
               "max": float('-inf'),
               "latencies": []
           }

       # Update statistics
       self._prediction_stats["count"] += 1
       self._prediction_stats["sum"] += prediction
       self._prediction_stats["sum_squared"] += prediction * prediction
       self._prediction_stats["min"] = min(self._prediction_stats["min"], prediction)
       self._prediction_stats["max"] = max(self._prediction_stats["max"], prediction)

       # Log every N predictions
       if self._prediction_stats["count"] % 100 == 0:
           mean = self._prediction_stats["sum"] / self._prediction_stats["count"]
           variance = (self._prediction_stats["sum_squared"] / self._prediction_stats["count"]) - (mean * mean)

           self.logger.info(
               f"Prediction statistics after {self._prediction_stats['count']} predictions: "
               f"mean={mean:.4f}, std={np.sqrt(variance):.4f}, "
               f"min={self._prediction_stats['min']:.4f}, "
               f"max={self._prediction_stats['max']:.4f}",
               source_module=self._source_module
           )
   ```

2. **Add Model Versioning Information**:
   ```python
   def _load_model_metadata(self) -> None:
       """Load and log model metadata."""
       try:
           metadata_path = self._model_path + ".metadata.json"
           import json
           with open(metadata_path, 'r') as f:
               metadata = json.load(f)

           self._model_version = metadata.get("version", "unknown")
           self._model_training_date = metadata.get("training_date", "unknown")
           self._model_metrics = metadata.get("metrics", {})

           self.logger.info(
               f"Loaded model metadata: version={self._model_version}, "
               f"trained={self._model_training_date}, "
               f"validation_accuracy={self._model_metrics.get('validation_accuracy', 'N/A')}",
               source_module=self._source_module
           )
       except FileNotFoundError:
           self.logger.warning(
               f"No metadata file found for model {self._model_path}",
               source_module=self._source_module
           )
       except Exception as e:
           self.logger.error(
               f"Error loading model metadata: {e}",
               source_module=self._source_module
           )
   ```

3. **Add Prediction Caching for Efficiency**:
   ```python
   def _get_prediction_cache_key(self, features: Dict[str, str]) -> str:
       """Generate a cache key from feature values."""
       # Sort keys to ensure consistent ordering
       sorted_items = sorted(features.items())
       return "_".join(f"{k}:{v}" for k, v in sorted_items)

   async def _get_cached_or_new_prediction(self, event: FeatureEvent) -> Dict[str, Any]:
       """Check cache before running prediction."""
       # Check if caching is enabled
       if not self._config.get("enable_caching", False):
           return await self._run_prediction_pipeline(event)

       # Generate cache key
       cache_key = self._get_prediction_cache_key(event.features)

       # Check if prediction is cached and not expired
       cache_ttl = self._config.get("cache_ttl_seconds", 5)  # Default 5 seconds TTL
       current_time = datetime.utcnow().timestamp()

       if hasattr(self, "_prediction_cache"):
           cached_item = self._prediction_cache.get(cache_key)
           if cached_item:
               prediction, timestamp = cached_item
               if current_time - timestamp < cache_ttl:
                   self.logger.debug(
                       "Using cached prediction",
                       source_module=self._source_module
                   )
                   return {"prediction": prediction, "cached": True}
       else:
           self._prediction_cache = {}

       # Run prediction and cache result
       result = await self._run_prediction_pipeline(event)
       if "prediction" in result:
           self._prediction_cache[cache_key] = (result["prediction"], current_time)

       return result
   ```

## Compliance Assessment

The module partially complies with the architectural requirements:

1. **Interface Implementation**: The implementation mostly conforms to the PredictionService interface defined in the interface definitions document, but lacks support for multiple model types and ensemble mechanisms.

2. **Functional Requirements**: Meets the core requirements of consuming feature events and publishing prediction events, but doesn't fully comply with FR-302 (multiple model types), FR-304 (feature preprocessing), FR-307 (ensemble logic), and FR-309-312 (retraining pipeline).

3. **Error Handling**: Good error management but lacks some recovery mechanisms.

4. **Performance**: Properly offloads inference to a separate process to avoid blocking the event loop, which should help meet NFR-501 latency requirements.

5. **Event Structure**: The prediction events appear to comply with the structure defined in the inter-module communication document.

## Follow-up Actions

- [ ] Implement support for scikit-learn and TensorFlow/PyTorch models (FR-302)
- [ ] Add model ensemble logic (FR-307)
- [ ] Implement feature preprocessing capabilities (FR-304)
- [ ] Clean up commented-out code and standardize logging approach
- [ ] Add configuration for model reloading without restart
- [ ] Improve handling of NaN and missing feature values
- [ ] Consider implementing model performance monitoring
- [ ] Implement model versioning and metadata tracking
- [ ] Add prediction caching for frequently used feature combinations
- [ ] Consider implementing the retraining pipeline or interfaces to it
