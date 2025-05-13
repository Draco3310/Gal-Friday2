# Advanced Ensemble Strategies for Prediction Service

This document outlines advanced ensemble strategies that could be implemented to enhance the prediction capabilities of Gal-Friday. These strategies go beyond the current simple averaging and weighted averaging methods.

## 1. Current Implementation

The current ensemble functionality includes:

- **Simple Average**: Equal weight to all model predictions
- **Weighted Average**: User-defined weights for each model

While effective for basic ensembling, more sophisticated strategies could improve prediction quality, particularly for financial time series data.

## 2. Proposed Advanced Strategies

### 2.1 Dynamic Time-Based Weighting

**Description**: Adjust model weights based on recent performance over configurable time windows.

**Algorithm**:
1. Track prediction accuracy for each model over multiple time windows (e.g., last hour, day, week)
2. Calculate performance metrics (e.g., RMSE, directional accuracy) for each window
3. Dynamically adjust weights based on relative performance

**Implementation Approach**:
```python
def dynamic_time_weighting(predictions, model_configs, performance_history):
    # Calculate recent performance for each model
    recent_performance = {}
    for model_id, perf_data in performance_history.items():
        # Calculate metrics for different time windows
        hourly_score = calculate_metric(perf_data['1h'])
        daily_score = calculate_metric(perf_data['1d'])
        weekly_score = calculate_metric(perf_data['1w'])

        # Combine scores with configurable importance
        recent_performance[model_id] = (
            hourly_score * 0.5 + daily_score * 0.3 + weekly_score * 0.2
        )

    # Normalize performance to create weights
    total_perf = sum(recent_performance.values())
    dynamic_weights = {
        model_id: perf / total_perf for model_id, perf in recent_performance.items()
    }

    # Apply weights to predictions
    weighted_sum = 0
    total_weight = 0

    for result, (config, _) in zip(predictions, model_configs):
        model_id = config.get('model_id')
        if 'prediction' in result and model_id in dynamic_weights:
            weight = dynamic_weights[model_id]
            weighted_sum += result['prediction'] * weight
            total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else None
```

**Configuration Example**:
```yaml
prediction_service:
  ensembles:
    - ensemble_id: "dynamic_time_ensemble"
      trading_pair: "BTC/USD"
      model_ids: ["xgb_model_v1", "lstm_model_v1"]
      strategy: "dynamic_time"
      time_windows: ["1h", "4h", "1d"]
      window_weights: [0.6, 0.3, 0.1]
      metric: "directional_accuracy"  # Alternative: "rmse", "sharpe"
```

### 2.2 Stacked Ensembling

**Description**: Use a meta-model that learns to combine predictions from base models.

**Algorithm**:
1. Base models make predictions independently
2. Their outputs become inputs to a "meta-model" that produces the final prediction
3. The meta-model is trained to optimize combined prediction quality

**Implementation Approach**:
```python
class StackedEnsemblePredictor(PredictorInterface):
    """Implements a stacked ensemble as a predictor."""

    def load_model(self) -> Any:
        # Load the meta-model (e.g., a simple regression model)
        try:
            return joblib.load(self.model_path)
        except Exception as e:
            self.logger.error(f"Error loading meta-model: {e}")
            return None

    def predict(self, base_model_predictions: np.ndarray) -> np.ndarray:
        """Take base model predictions and produce final prediction."""
        if self.model is None:
            raise TypeError("Meta-model not loaded")

        # base_model_predictions shape: [n_samples, n_base_models]
        return self.model.predict(base_model_predictions)
```

**Integration in Prediction Service**:
```python
async def _run_stacked_ensemble_pipeline(self, event, ensemble_config):
    # Get base model predictions first
    base_predictions = []
    for model_config in self._get_base_model_configs(ensemble_config):
        result = await self._get_model_prediction(event, model_config)
        if 'prediction' in result:
            base_predictions.append(result['prediction'])

    # Convert to features for meta-model
    meta_features = np.array(base_predictions).reshape(1, -1)

    # Get meta-model config
    meta_model_id = ensemble_config.get('meta_model_id')
    meta_model_config = self._get_model_config(meta_model_id)

    # Run meta-model prediction
    meta_result = await self._run_in_executor(
        self._process_pool_executor,
        _run_meta_model_inference,
        meta_model_config,
        meta_features
    )

    # Create and publish final prediction
    if 'prediction' in meta_result:
        prediction_event = self._create_prediction_event(
            event,
            ensemble_config.get('output_model_id', ensemble_config.get('ensemble_id')),
            ensemble_config.get('prediction_target'),
            meta_result['prediction'],
            {'ensemble_strategy': 'stacked'}
        )
        await self._publish_prediction(prediction_event)
```

**Configuration Example**:
```yaml
prediction_service:
  ensembles:
    - ensemble_id: "stacked_ensemble_v1"
      trading_pair: "BTC/USD"
      model_ids: ["xgb_model_v1", "lstm_model_v1", "rf_model_v1"]
      strategy: "stacked"
      meta_model_id: "meta_model_v1"
      output_model_id: "stacked_ensemble_v1"
      prediction_target: "price_movement_5min"

  models:
    # Base models definitions...

    # Meta-model definition
    - model_id: "meta_model_v1"
      model_path: "models/meta/stacking_model.joblib"
      model_type: "sklearn"
```

### 2.3 Market Regime-Based Ensemble

**Description**: Switch between different model weights based on identified market regimes.

**Algorithm**:
1. Identify current market regime (e.g., trending, range-bound, volatile)
2. Use pre-configured weights for each regime type
3. Apply the appropriate weight set based on the current regime

**Implementation Approach**:
```python
def regime_based_ensemble(predictions, model_configs, feature_data, regime_weights):
    # Determine current market regime
    regime = identify_market_regime(feature_data)

    # Get weights for the current regime
    current_weights = regime_weights.get(regime, {})

    # Apply weights to predictions
    weighted_sum = 0
    total_weight = 0

    for result, (config, _) in zip(predictions, model_configs):
        model_id = config.get('model_id')
        if 'prediction' in result and model_id in current_weights:
            weight = current_weights[model_id]
            weighted_sum += result['prediction'] * weight
            total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else None

def identify_market_regime(feature_data):
    """Identify the current market regime based on features."""
    volatility = calculate_volatility(feature_data)
    trend_strength = calculate_trend_strength(feature_data)

    if volatility > 0.8:
        return "high_volatility"
    elif trend_strength > 0.7:
        return "trending"
    else:
        return "range_bound"
```

**Configuration Example**:
```yaml
prediction_service:
  ensembles:
    - ensemble_id: "regime_ensemble_v1"
      trading_pair: "BTC/USD"
      model_ids: ["xgb_model_v1", "lstm_model_v1", "mean_reversion_model_v1"]
      strategy: "regime_based"
      regime_weights:
        trending:
          "xgb_model_v1": 0.6
          "lstm_model_v1": 0.4
          "mean_reversion_model_v1": 0.0
        range_bound:
          "xgb_model_v1": 0.2
          "lstm_model_v1": 0.1
          "mean_reversion_model_v1": 0.7
        high_volatility:
          "xgb_model_v1": 0.4
          "lstm_model_v1": 0.5
          "mean_reversion_model_v1": 0.1
      regime_detection:
        volatility_window: 20
        trend_strength_window: 50
        volatility_threshold: 0.08
        trend_threshold: 0.6
```

### 2.4 Bayesian Model Averaging

**Description**: Use Bayesian inference to determine the optimal combination of model outputs.

**Algorithm**:
1. Assign prior probabilities to each model
2. Update model probabilities using Bayes' theorem based on observed performance
3. Compute a weighted average using posterior probabilities

**Implementation Approach**:
```python
def bayesian_model_averaging(predictions, model_configs, prior_probs, likelihood_func):
    # Calculate posterior probability for each model
    posterior_probs = {}

    for i, (result, (config, _)) in enumerate(zip(predictions, model_configs)):
        model_id = config.get('model_id')
        if 'prediction' in result and model_id in prior_probs:
            # Calculate likelihood based on recent performance
            likelihood = likelihood_func(model_id)

            # Update posterior: P(model|data) ∝ P(data|model) * P(model)
            posterior_probs[model_id] = likelihood * prior_probs[model_id]

    # Normalize posterior probabilities
    total = sum(posterior_probs.values())
    if total > 0:
        posterior_probs = {k: v/total for k, v in posterior_probs.items()}

    # Calculate weighted prediction using posterior probabilities
    weighted_sum = 0
    for result, (config, _) in zip(predictions, model_configs):
        model_id = config.get('model_id')
        if 'prediction' in result and model_id in posterior_probs:
            weighted_sum += result['prediction'] * posterior_probs[model_id]

    return weighted_sum
```

**Configuration Example**:
```yaml
prediction_service:
  ensembles:
    - ensemble_id: "bayesian_ensemble_v1"
      trading_pair: "BTC/USD"
      model_ids: ["xgb_model_v1", "lstm_model_v1", "rf_model_v1"]
      strategy: "bayesian"
      prior_probabilities:
        "xgb_model_v1": 0.4
        "lstm_model_v1": 0.4
        "rf_model_v1": 0.2
      likelihood_method: "inverse_rmse"  # How to calculate likelihood
      update_frequency: "1h"  # How often to update posterior probabilities
```

### 2.5 Confidence-Weighted Ensemble

**Description**: Weight models based on their self-reported confidence in each prediction.

**Algorithm**:
1. Each model provides both a prediction and a confidence score
2. Weight predictions proportionally to the confidence scores
3. Normalize by total confidence

**Implementation Approach**:
```python
def confidence_weighted_ensemble(predictions_with_confidence):
    """Combine predictions based on model's confidence in each prediction."""
    weighted_sum = 0
    total_confidence = 0

    for prediction in predictions_with_confidence:
        if 'prediction' in prediction and 'confidence' in prediction:
            confidence = float(prediction['confidence'])
            weighted_sum += prediction['prediction'] * confidence
            total_confidence += confidence

    return weighted_sum / total_confidence if total_confidence > 0 else None
```

**Modifications to Predictor Interface**:
```python
class ConfidencePredictor(PredictorInterface):
    """Predictor that reports confidence along with predictions."""

    def predict_with_confidence(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate predictions and confidence scores."""
        if self.model is None:
            raise TypeError("Model has not been loaded")

        # Get predictions
        predictions = self.model.predict(features)

        # For models that provide probability estimates
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(features)
            # Use max probability as confidence
            confidences = np.max(probas, axis=1)
        else:
            # Generate synthetic confidence based on feature quality
            confidences = self._estimate_confidence(features)

        return {
            "predictions": predictions,
            "confidences": confidences
        }
```

**Configuration Example**:
```yaml
prediction_service:
  ensembles:
    - ensemble_id: "confidence_ensemble_v1"
      trading_pair: "BTC/USD"
      model_ids: ["xgb_model_v1", "lstm_model_v1", "rf_model_v1"]
      strategy: "confidence_weighted"
      confidence_floor: 0.2  # Minimum confidence to consider
      normalize_confidences: true
```

## 3. Implementation Strategy

To add these advanced ensemble strategies:

1. **Define Strategy Interfaces**:
   ```python
   from abc import ABC, abstractmethod
   from typing import List, Dict, Any, Optional
   import numpy as np

   class EnsembleStrategy(ABC):
       """Base interface for ensemble strategies."""

       @abstractmethod
       def combine(
           self,
           predictions: List[Dict[str, Any]],
           model_configs: List[Dict[str, Any]],
           context: Dict[str, Any]
       ) -> Optional[float]:
           """Combine predictions using the strategy."""
           pass
   ```

2. **Implement Each Strategy**:
   ```python
   class DynamicTimeWeightingStrategy(EnsembleStrategy):
       """Dynamic time-based weighting ensemble strategy."""

       def combine(self, predictions, model_configs, context):
           # Implementation as described above
           pass
   ```

3. **Strategy Factory**:
   ```python
   def create_ensemble_strategy(strategy_type: str, config: Dict[str, Any]) -> EnsembleStrategy:
       """Create appropriate ensemble strategy based on configuration."""
       if strategy_type == "average":
           return AverageStrategy()
       elif strategy_type == "weighted":
           return WeightedStrategy(config.get("weights", {}))
       elif strategy_type == "dynamic_time":
           return DynamicTimeWeightingStrategy(
               config.get("time_windows", ["1h", "1d"]),
               config.get("window_weights", [0.7, 0.3]),
               config.get("metric", "directional_accuracy")
           )
       # Add more strategies here
       else:
           raise ValueError(f"Unknown ensemble strategy: {strategy_type}")
   ```

4. **Update Prediction Service**:
   ```python
   def _combine_predictions(self, results, model_configs, strategy_type, additional_config=None):
       """Combine predictions using the specified strategy."""
       context = {
           "feature_data": self._current_feature_data,
           "performance_history": self._model_performance_tracker.get_history(),
           "current_time": datetime.now(timezone.utc)
       }

       strategy = create_ensemble_strategy(strategy_type, additional_config or {})
       return strategy.combine(results, model_configs, context)
   ```

## 4. Performance and Quality Metrics

To evaluate and compare ensemble strategies:

### 4.1 Prediction Accuracy Metrics

- **RMSE (Root Mean Squared Error)**: For regression tasks
- **Directional Accuracy**: For binary movement prediction
- **Profit & Loss**: Ultimate measure of trading strategy performance

### 4.2 Strategy-Specific Metrics

- **Dynamic Time Weighting**: Monitor weight adjustment frequency
- **Stacked Ensemble**: Measure meta-model vs. base model performance
- **Regime-Based**: Track regime identification accuracy
- **Bayesian**: Monitor posterior probability convergence

### 4.3 Resource Usage Metrics

- **Computation Time**: Ensure strategies can run within the required latency
- **Memory Usage**: Track additional memory requirements
- **Scaling Behavior**: How performance scales with number of models

## 5. Phased Implementation Plan

1. **Phase 1**: Implement Confidence-Weighted Ensemble
   - Requires minimal changes to existing architecture
   - Provides immediate value by leveraging model confidence

2. **Phase 2**: Implement Dynamic Time-Based Weighting
   - Add performance tracking infrastructure
   - Integrate with existing monitoring systems

3. **Phase 3**: Implement Regime-Based Ensemble
   - Develop market regime identification algorithms
   - Create regime-specific model weights

4. **Phase 4**: Implement Stacked Ensembling
   - Develop training infrastructure for meta-models
   - Create efficient meta-model inference pipeline

5. **Phase 5**: Implement Bayesian Model Averaging
   - Develop Bayesian framework for model evaluation
   - Implement efficient posterior probability updates

## 6. Conclusion

These advanced ensemble strategies can significantly improve prediction quality by:

- Adapting to changing market conditions
- Leveraging relative strengths of different model types
- Incorporating uncertainty and confidence measures
- Optimizing model combinations based on historical performance

By implementing these strategies, Gal-Friday can achieve more robust and accurate predictions, which directly translates to better trading performance.
