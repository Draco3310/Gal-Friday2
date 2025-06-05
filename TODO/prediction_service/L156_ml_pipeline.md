# Task: Implement ML pipeline for price prediction with feature engineering.

### 1. Context
- **File:** `gal_friday/prediction_service.py`
- **Line:** `156`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing ML pipeline for price prediction with feature engineering.

### 2. Problem Statement
Without proper ML pipeline for price prediction, the system cannot generate accurate trading signals or make informed decisions based on market data patterns. This prevents the implementation of sophisticated trading strategies and limits the system's predictive capabilities.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Feature Engineering Pipeline:** Comprehensive feature extraction and transformation
2. **Build Model Training Framework:** Automated model training and validation
3. **Implement Prediction Engine:** Real-time prediction generation and serving
4. **Add Model Management:** Version control, deployment, and monitoring
5. **Create Performance Evaluation:** Model performance tracking and optimization
6. **Build Data Pipeline:** Efficient data preprocessing and feature storage

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging

class ModelType(str, Enum):
    """Supported ML model types"""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    LINEAR_REGRESSION = "linear_regression"

@dataclass
class PredictionRequest:
    """Request for price prediction"""
    symbol: str
    features: Dict[str, float]
    prediction_horizon: int  # minutes ahead
    confidence_level: float = 0.95

@dataclass
class PredictionResult:
    """Result of price prediction"""
    symbol: str
    predicted_price: float
    confidence_interval: Tuple[float, float]
    prediction_timestamp: datetime
    model_version: str
    feature_importance: Dict[str, float]

class MLPredictionPipeline:
    """Enterprise-grade ML pipeline for price prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model management
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_columns: List[str] = []
        
        # Performance tracking
        self.prediction_stats = {
            'predictions_made': 0,
            'accuracy_score': 0.0,
            'last_training_time': None
        }
    
    async def train_model(self, symbol: str, training_data: pd.DataFrame) -> None:
        """
        Train ML model for price prediction
        Replace TODO with comprehensive ML pipeline
        """
        
        try:
            self.logger.info(f"Training ML model for {symbol}")
            
            # Feature engineering
            features_df = await self._engineer_features(training_data)
            
            # Prepare training data
            X, y = self._prepare_training_data(features_df)
            
            # Train model
            model = self._create_model(self.config.get('model_type', ModelType.RANDOM_FOREST))
            model.fit(X, y)
            
            # Create and fit scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Store model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.feature_columns = list(features_df.columns)
            
            # Evaluate model performance
            train_score = model.score(X_scaled, y)
            
            self.logger.info(f"Model trained for {symbol} with score: {train_score:.4f}")
            self.prediction_stats['last_training_time'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to train model for {symbol}: {e}")
            raise MLPipelineError(f"Model training failed: {e}")
    
    async def predict_price(self, request: PredictionRequest) -> PredictionResult:
        """Generate price prediction"""
        
        try:
            symbol = request.symbol
            
            if symbol not in self.models:
                raise MLPipelineError(f"No trained model available for {symbol}")
            
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            
            # Prepare features
            feature_vector = self._prepare_features(request.features)
            feature_vector_scaled = scaler.transform([feature_vector])
            
            # Make prediction
            predicted_price = model.predict(feature_vector_scaled)[0]
            
            # Calculate confidence interval (simplified)
            prediction_std = self._estimate_prediction_uncertainty(model, feature_vector_scaled)
            confidence_margin = 1.96 * prediction_std  # 95% confidence
            
            confidence_interval = (
                predicted_price - confidence_margin,
                predicted_price + confidence_margin
            )
            
            # Get feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                for i, importance in enumerate(model.feature_importances_):
                    feature_importance[self.feature_columns[i]] = importance
            
            result = PredictionResult(
                symbol=symbol,
                predicted_price=predicted_price,
                confidence_interval=confidence_interval,
                prediction_timestamp=datetime.now(),
                model_version="1.0",
                feature_importance=feature_importance
            )
            
            self.prediction_stats['predictions_made'] += 1
            
            self.logger.debug(f"Predicted price for {symbol}: ${predicted_price:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {request.symbol}: {e}")
            raise MLPipelineError(f"Prediction failed: {e}")
    
    async def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw market data"""
        
        features_df = data.copy()
        
        # Technical indicators
        features_df['sma_20'] = data['close'].rolling(window=20).mean()
        features_df['sma_50'] = data['close'].rolling(window=50).mean()
        features_df['rsi'] = self._calculate_rsi(data['close'])
        features_df['volatility'] = data['close'].rolling(window=20).std()
        
        # Price-based features
        features_df['price_change'] = data['close'].pct_change()
        features_df['high_low_ratio'] = data['high'] / data['low']
        features_df['volume_sma'] = data['volume'].rolling(window=20).mean()
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features_df[f'close_lag_{lag}'] = data['close'].shift(lag)
            features_df[f'volume_lag_{lag}'] = data['volume'].shift(lag)
        
        # Drop rows with NaN values
        features_df = features_df.dropna()
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training"""
        
        # Use all columns except the target
        feature_columns = [col for col in features_df.columns if col not in ['close', 'open', 'high', 'low']]
        
        X = features_df[feature_columns].values
        y = features_df['close'].values
        
        return X, y
    
    def _create_model(self, model_type: ModelType) -> Any:
        """Create ML model based on type"""
        
        if model_type == ModelType.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == ModelType.LINEAR_REGRESSION:
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _prepare_features(self, features: Dict[str, float]) -> List[float]:
        """Prepare feature vector from feature dictionary"""
        
        feature_vector = []
        for column in self.feature_columns:
            if column in features:
                feature_vector.append(features[column])
            else:
                feature_vector.append(0.0)  # Default value for missing features
        
        return feature_vector
    
    def _estimate_prediction_uncertainty(self, model: Any, features: np.ndarray) -> float:
        """Estimate prediction uncertainty"""
        
        # Simplified uncertainty estimation
        if hasattr(model, 'estimators_'):
            # For ensemble models, use prediction variance
            predictions = [estimator.predict(features)[0] for estimator in model.estimators_]
            return np.std(predictions)
        else:
            # Default uncertainty
            return 0.01  # 1% of price as default uncertainty
    
    def get_model_performance(self, symbol: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        
        return {
            'symbol': symbol,
            'model_available': symbol in self.models,
            'predictions_made': self.prediction_stats['predictions_made'],
            'last_training_time': self.prediction_stats['last_training_time'],
            'feature_count': len(self.feature_columns)
        }

class MLPipelineError(Exception):
    """Exception raised for ML pipeline errors"""
    pass
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Robust handling of missing data; graceful degradation when models fail; comprehensive error logging
- **Configuration:** Configurable model parameters; feature engineering options; training schedules
- **Testing:** Unit tests for feature engineering; model validation tests; performance benchmarking
- **Dependencies:** Scikit-learn for ML models; pandas for data manipulation; feature engineering libraries

### 4. Acceptance Criteria
- [ ] Feature engineering pipeline creates comprehensive technical indicators and market features
- [ ] Model training framework supports multiple ML algorithms with automated validation
- [ ] Prediction engine generates real-time price predictions with confidence intervals
- [ ] Model management includes version control, deployment, and performance monitoring
- [ ] Performance evaluation tracks prediction accuracy and model drift
- [ ] Data pipeline efficiently processes market data for feature extraction
- [ ] Uncertainty estimation provides confidence intervals for predictions
- [ ] Feature importance analysis identifies key predictive factors
- [ ] Model retraining automation maintains prediction accuracy over time
- [ ] TODO placeholder is completely replaced with production-ready implementation 