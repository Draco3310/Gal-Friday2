# Adding New Model Types - Extension Guide

This document outlines the process for extending Gal-Friday to support additional machine learning model types beyond the currently implemented XGBoost and scikit-learn models.

## 1. Architecture Overview

The prediction system is designed with extensibility in mind, using a modular approach:

- **Interface Layer**: `PredictorInterface` defines the contract all model implementations must follow
- **Prediction Service**: Handles model loading, prediction coordination, and ensemble methods
- **Model Implementations**: Concrete classes for specific model types (XGBoost, scikit-learn, etc.)
- **Configuration**: Flexible configuration format for defining models and their parameters

## 2. Adding a New Model Type

### 2.1 Implement the PredictorInterface

Create a new class that implements `PredictorInterface` for your model type:

```python
# src/gal_friday/predictors/tensorflow_predictor.py
import numpy as np
import tensorflow as tf
from typing import List, Optional, Dict, Any
from ..interfaces.predictor_interface import PredictorInterface

class TensorFlowPredictor(PredictorInterface):
    """Implementation of PredictorInterface for TensorFlow models."""

    def load_model(self) -> Any:
        """Load the TensorFlow model from the specified path."""
        try:
            return tf.saved_model.load(self.model_path)
        except (FileNotFoundError, IOError) as e:
            self.logger.error(f"Error loading TensorFlow model: {e}")
            return None

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions using the TensorFlow model."""
        if self.model is None:
            raise TypeError("Model has not been loaded")

        # For TensorFlow models that use the SavedModel format
        predict_fn = self.model.signatures["serving_default"]
        tensor_input = tf.convert_to_tensor(features, dtype=tf.float32)
        result = predict_fn(tensor_input)

        # Extract prediction from result - adjust based on model output format
        prediction_key = list(result.keys())[0]
        return result[prediction_key].numpy()

    @property
    def expected_feature_names(self) -> Optional[List[str]]:
        """Return the list of feature names expected by the model."""
        # Get feature names from config
        model_features = self.config.get("model_feature_names")
        if model_features:
            return model_features

        # Alternatively, try to extract from model metadata if available
        # This is model-specific and may need customization
        return None
```

### 2.2 Update Import Registry

Update the import registry to make the new predictor available:

```python
# src/gal_friday/predictors/__init__.py
from .xgboost_predictor import XGBoostPredictor
from .sklearn_predictor import SklearnPredictor
from .tensorflow_predictor import TensorFlowPredictor

__all__ = ["XGBoostPredictor", "SklearnPredictor", "TensorFlowPredictor"]
```

### 2.3 Update Prediction Service Model Loading Logic

If needed, update the model creation logic in `prediction_service.py`:

```python
def _run_inference_task(model_config, features_dict):
    # ...existing code...

    # Create appropriate predictor based on model type
    try:
        if model_type == "xgboost":
            predictor = XGBoostPredictor(model_path, model_id, model_config)
        elif model_type == "sklearn":
            predictor = SklearnPredictor(model_path, model_id, model_config)
        elif model_type == "tensorflow":
            predictor = TensorFlowPredictor(model_path, model_id, model_config)
        # Add new model type here
        elif model_type == "pytorch":
            predictor = PyTorchPredictor(model_path, model_id, model_config)
        else:
            return {"error": f"Unsupported model type: {model_type}"}
    # ...existing code...
```

## 3. Configuration Examples

### 3.1 TensorFlow Model Configuration

```yaml
prediction_service:
  models:
    - model_id: "tf_model_v1"
      trading_pair: "BTC/USD"
      model_path: "models/tensorflow/btc_usd_v1"  # SavedModel directory path
      model_type: "tensorflow"
      model_feature_names: ["mid_price", "volume", "rsi_14", "macd", "bb_width"]
      prediction_target: "price_movement_5min"
      preprocessing:
        scaler_path: "models/tensorflow/btc_usd_scaler.joblib"
        max_nan_percentage: 15.0
```

### 3.2 PyTorch Model Configuration

```yaml
prediction_service:
  models:
    - model_id: "pytorch_model_v1"
      trading_pair: "ETH/USD"
      model_path: "models/pytorch/eth_usd_v1.pt"
      model_type: "pytorch"
      model_feature_names: ["mid_price", "volume", "rsi_14", "macd", "bb_width"]
      prediction_target: "price_movement_5min"
      preprocessing:
        scaler_path: "models/pytorch/eth_usd_scaler.joblib"
        max_nan_percentage: 15.0
      # Model-specific parameters
      model_params:
        input_size: 5
        output_size: 1
```

## 4. Ensemble Considerations

### 4.1 Normalizing Outputs Across Model Types

Different model types may produce outputs in different formats and scales. When creating ensembles with multiple model types, consider these approaches:

1. **Standardize Output Range**: Ensure all models output values in the same range (e.g., 0-1 for binary classification)
2. **Output Transformation**: Apply normalization functions in the predictor implementation
3. **Post-Processing**: Add a strategy-specific normalization step in the ensemble processing

Example ensemble configuration mixing model types:

```yaml
prediction_service:
  ensembles:
    - ensemble_id: "mixed_model_ensemble_v1"
      trading_pair: "BTC/USD"
      model_ids: ["xgb_model_v1", "tf_model_v1", "pytorch_model_v1"]
      strategy: "weighted"
      weights:
        "xgb_model_v1": 0.4
        "tf_model_v1": 0.4
        "pytorch_model_v1": 0.2
      output_model_id: "mixed_model_ensemble_v1"
      prediction_target: "price_movement_5min"
      # Optional output normalization
      output_processing: "softmax"  # Can be: "none", "softmax", "minmax", "custom"
```

## 5. Testing New Model Implementations

When adding a new model type, create comprehensive tests:

1. **Unit Tests**: Test the new predictor class
2. **Integration Tests**: Test the prediction service with the new model type
3. **Ensemble Tests**: Test ensemble combinations including the new model type

Example test structure:

```python
# tests/unit/predictors/test_tensorflow_predictor.py

def test_tensorflow_model_loading():
    # Test model loading functionality

def test_tensorflow_prediction():
    # Test prediction functionality

def test_tensorflow_feature_extraction():
    # Test feature name extraction

def test_tensorflow_error_handling():
    # Test error handling
```

## 6. Performance Considerations

Different model types have varying performance characteristics:

- **Memory Usage**: Deep learning models (TensorFlow, PyTorch) typically use more memory
- **Loading Time**: Large models may take longer to load
- **Inference Speed**: Consider the trade-off between accuracy and speed

Implement appropriate optimizations:

```python
class TensorFlowPredictor(PredictorInterface):
    def __init__(self, model_path, model_id, config):
        super().__init__(model_path, model_id, config)

        # Optional GPU configuration
        if config.get("use_gpu", False):
            # Configure TensorFlow to use GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                tf.config.experimental.set_memory_growth(gpus[0], True)
        else:
            # Restrict TensorFlow to CPU only
            tf.config.set_visible_devices([], 'GPU')
```

## 7. Dependencies Management

Update the project dependencies to include the required packages:

**pyproject.toml**:
```toml
[tool.poetry.dependencies]
python = "^3.10"
xgboost = "^1.7.3"
scikit-learn = "^1.2.0"
tensorflow = {version = "^2.11.0", optional = true}
torch = {version = "^1.13.1", optional = true}

[tool.poetry.extras]
tensorflow = ["tensorflow"]
pytorch = ["torch"]
all = ["tensorflow", "torch"]
```

This allows users to install only the dependencies they need:

```bash
# Install with TensorFlow support
pip install -e ".[tensorflow]"

# Install with PyTorch support
pip install -e ".[pytorch]"

# Install with all model types
pip install -e ".[all]"
```

## 8. Common Challenges and Solutions

### 8.1 Different Input Formats

**Challenge**: Different model types expect different input formats.

**Solution**: Standardize inputs in the predictor implementation:

```python
def predict(self, features: np.ndarray) -> np.ndarray:
    # Tensorflow may expect specific shape
    if features.ndim == 2:
        # For batch prediction
        tensor_input = tf.convert_to_tensor(features, dtype=tf.float32)
    else:
        # For single prediction, add batch dimension
        tensor_input = tf.convert_to_tensor(features.reshape(1, -1), dtype=tf.float32)
```

### 8.2 Model-Specific Preprocessing

**Challenge**: Each model type may require specific preprocessing steps.

**Solution**: Add model-specific preprocessing in the predictor, parametrized by configuration:

```python
def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
    # Apply model-specific preprocessing
    preprocessing_config = self.config.get("preprocessing", {})

    # Apply specific transformations
    if preprocessing_config.get("normalize", False):
        features = (features - self.model.normalization_mean) / self.model.normalization_std

    return features
```

### 8.3 Different Output Formats

**Challenge**: Models produce different output formats.

**Solution**: Standardize outputs in the predictor:

```python
def predict(self, features: np.ndarray) -> np.ndarray:
    # Get raw prediction
    raw_output = self.model.predict(features)

    # Extract and standardize output
    output_format = self.config.get("output_format", "raw")

    if output_format == "probability":
        # Convert logits to probability if needed
        return np.array([self._convert_to_probability(o) for o in raw_output])

    return raw_output
```

## 9. Upgrading Existing Systems

When deploying a new model type to production:

1. **Gradual Rollout**: Start with low ensemble weights for the new model type
2. **A/B Testing**: Compare performance against existing model types
3. **Monitoring**: Track resource usage and prediction quality
4. **Fallback Mechanism**: Implement a fallback to existing models if the new one fails

By following this guide, you can extend Gal-Friday to support additional model types while maintaining the system's reliability and performance.
