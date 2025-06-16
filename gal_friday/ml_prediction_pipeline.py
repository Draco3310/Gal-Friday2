# ML Prediction Pipeline Module
"""Enterprise-grade ML pipeline for price prediction with comprehensive feature engineering.

This module provides infrastructure for training ML models, feature engineering,
model lifecycle management, and performance evaluation for trading predictions.
"""

import asyncio
import pickle
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from gal_friday.logger_service import LoggerService


class ModelType(str, Enum):
    """Supported ML model types."""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    LINEAR_REGRESSION = "linear_regression"


class PipelineStage(str, Enum):
    """ML pipeline execution stages."""
    DATA_VALIDATION = "data_validation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_DEPLOYMENT = "model_deployment"


@dataclass
class PredictionRequest:
    """Request for price prediction."""
    symbol: str
    features: Dict[str, float]
    prediction_horizon: int  # minutes ahead
    confidence_level: float = 0.95
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PredictionResult:
    """Result of price prediction."""
    symbol: str
    predicted_price: float
    confidence_interval: Tuple[float, float]
    prediction_timestamp: datetime
    model_version: str
    feature_importance: Dict[str, float]
    request_id: str
    model_id: str
    accuracy_metrics: Optional[Dict[str, float]] = None


@dataclass
class ModelTrainingConfig:
    """Configuration for model training."""
    model_type: ModelType
    target_column: str = "close"
    feature_columns: Optional[List[str]] = None
    training_window_days: int = 30
    validation_split: float = 0.2
    cv_folds: int = 5
    hyperparameters: Dict[str, Any] = field(default_factory=dict[str, Any])
    performance_threshold: float = 0.0
    retrain_interval_hours: int = 24


@dataclass
class ModelPerformanceMetrics:
    """Model performance tracking metrics."""
    model_id: str
    timestamp: datetime
    mse: float
    mae: float
    r2_score: float
    accuracy_score: float
    prediction_count: int
    training_time_seconds: float
    feature_count: int
    validation_score: float


class MLPipelineError(Exception):
    """Base exception for ML pipeline errors."""
    pass


class FeatureEngineeringError(MLPipelineError):
    """Exception raised for feature engineering errors."""
    pass


class ModelTrainingError(MLPipelineError):
    """Exception raised for model training errors."""
    pass


class ModelValidationError(MLPipelineError):
    """Exception raised for model validation errors."""
    pass


class FeatureEngineer:
    """Advanced feature engineering for financial time series data."""
    
    def __init__(self, config: Dict[str, Any], logger: LoggerService) -> None:
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features from raw market data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
            
        Raises:
            FeatureEngineeringError: If feature engineering fails
        """
        try:
            self.logger.info(
                "Starting feature engineering for %(rows)d data points",
                source_module=self._source_module,
                context={"rows": len(data)}
            )
            
            features_df = data.copy()
            
            # Technical indicators
            features_df = self._add_technical_indicators(features_df)
            
            # Price-based features
            features_df = self._add_price_features(features_df)
            
            # Volume features
            features_df = self._add_volume_features(features_df)
            
            # Lag features
            features_df = self._add_lag_features(features_df)
            
            # Statistical features
            features_df = self._add_statistical_features(features_df)
            
            # Time-based features
            features_df = self._add_time_features(features_df)
            
            # Clean and validate features
            features_df = self._clean_features(features_df)
            
            self.logger.info(
                "Feature engineering completed. %(features)d features created",
                source_module=self._source_module,
                context={"features": len(features_df.columns)}
            )
            
            return features_df
            
        except Exception as e:
            self.logger.error(
                "Feature engineering failed: %(error)s",
                source_module=self._source_module,
                context={"error": str(e)},
                exc_info=True
            )
            raise FeatureEngineeringError(f"Feature engineering failed: {e}") from e
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # Bollinger Bands
        df['bb_upper'] = df['sma_20'] + (df['close'].rolling(20).std() * 2)
        df['bb_lower'] = df['sma_20'] - (df['close'].rolling(20).std() * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volatility
        for window in [10, 20, 30]:
            df[f'volatility_{window}'] = df['close'].rolling(window=window).std()
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Price changes
        for period in [1, 2, 3, 5]:
            df[f'price_change_{period}'] = df['close'].pct_change(periods=period)
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['high_close_ratio'] = df['high'] / df['close']
        df['low_close_ratio'] = df['low'] / df['close']
        
        # Price position in range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        if 'volume' not in df.columns:
            return df
            
        # Volume moving averages
        for window in [5, 10, 20]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
        
        # Volume ratios
        df['volume_ratio_5'] = df['volume'] / df['volume_sma_5']
        df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']
        
        # Price-volume features
        df['price_volume'] = df['close'] * df['volume']
        df['vwap'] = (df['price_volume'].rolling(20).sum() / 
                      df['volume'].rolling(20).sum())
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        lag_periods = [1, 2, 3, 5, 10]
        
        for lag in lag_periods:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df.get('volume', pd.Series()).shift(lag)
            df[f'rsi_lag_{lag}'] = df.get('rsi', pd.Series()).shift(lag)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        windows = [5, 10, 20]
        
        for window in windows:
            # Rolling statistics
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'close_skew_{window}'] = df['close'].rolling(window).skew()
            df[f'close_kurt_{window}'] = df['close'].rolling(window).kurt()
            
            # Percentile features
            df[f'close_percentile_{window}'] = (
                df['close'].rolling(window).rank() / window
            )
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
            
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series[Any], window: int = 14) -> pd.Series[Any]:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series[Any], 
                       fast: int = 12, slow: int = 26, signal: int = 9
                       ) -> Tuple[pd.Series[Any], pd.Series[Any]]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        
        return macd, macd_signal
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate engineered features."""
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns with too many NaN values
        nan_threshold = self.config.get('nan_threshold', 0.5)
        df = df.dropna(thresh=int(len(df) * (1 - nan_threshold)), axis=1)
        
        # Forward fill remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df


class ModelTrainer:
    """Model training and validation component."""
    
    def __init__(self, config: Dict[str, Any], logger: LoggerService) -> None:
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__
        
    async def train_model(self, 
                         training_data: pd.DataFrame,
                         training_config: ModelTrainingConfig,
                         symbol: str) -> Dict[str, Any]:
        """Train ML model with comprehensive validation.
        
        Args:
            training_data: DataFrame with features and target
            training_config: Training configuration
            symbol: Trading symbol being trained for
            
        Returns:
            Dictionary containing trained model, scaler, and metrics
            
        Raises:
            ModelTrainingError: If training fails
        """
        try:
            self.logger.info(
                "Starting model training for %(symbol)s with %(model_type)s",
                source_module=self._source_module,
                context={
                    "symbol": symbol,
                    "model_type": training_config.model_type.value,
                    "data_points": len(training_data)
                }
            )
            
            start_time = datetime.utcnow()
            
            # Prepare training data
            X, y = self._prepare_training_data(training_data, training_config)
            
            # Create and train model
            model = self._create_model(training_config.model_type, 
                                     training_config.hyperparameters)
            
            # Create and fit scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform cross-validation
            cv_scores = self._perform_cross_validation(model, X_scaled, y, 
                                                     training_config.cv_folds)
            
            # Train final model
            model.fit(X_scaled, y)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(
                model, scaler, X, y, symbol, start_time
            )
            
            # Validate performance threshold
            if metrics.validation_score < training_config.performance_threshold:
                raise ModelValidationError(
                    f"Model performance {metrics.validation_score:.4f} below "
                    f"threshold {training_config.performance_threshold:.4f}"
                )
            
            result = {
                "model": model,
                "scaler": scaler,
                "metrics": metrics,
                "feature_names": list[Any](X.columns),
                "cv_scores": cv_scores,
                "training_config": training_config,
                "model_version": f"{symbol}_{training_config.model_type.value}_{int(start_time.timestamp())}"
            }
            
            self.logger.info(
                "Model training completed for %(symbol)s. Validation score: %(score).4f",
                source_module=self._source_module,
                context={
                    "symbol": symbol,
                    "score": metrics.validation_score,
                    "training_time": metrics.training_time_seconds
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Model training failed for %(symbol)s: %(error)s",
                source_module=self._source_module,
                context={"symbol": symbol, "error": str(e)},
                exc_info=True
            )
            raise ModelTrainingError(f"Model training failed: {e}") from e
    
    def _prepare_training_data(self, 
                              data: pd.DataFrame, 
                              config: ModelTrainingConfig
                              ) -> Tuple[pd.DataFrame, pd.Series[Any]]:
        """Prepare features and target for training."""
        # Select feature columns
        if config.feature_columns:
            feature_cols = [col for col in config.feature_columns if col in data.columns]
        else:
            # Exclude target and timestamp columns
            exclude_cols = {config.target_column, 'timestamp', 'date', 'time'}
            feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_cols].copy()
        y = data[config.target_column].copy()
        
        # Remove rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def _create_model(self, model_type: ModelType, hyperparameters: Dict[str, Any]) -> Any:
        """Create ML model based on type and hyperparameters."""
        if model_type == ModelType.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=hyperparameters.get('n_estimators', 100),
                max_depth=hyperparameters.get('max_depth', 10),
                random_state=hyperparameters.get('random_state', 42),
                n_jobs=hyperparameters.get('n_jobs', -1)
            )
        elif model_type == ModelType.LINEAR_REGRESSION:
            return LinearRegression()
        elif model_type == ModelType.XGBOOST:
            try:
                import xgboost as xgb
                return xgb.XGBRegressor(
                    n_estimators=hyperparameters.get('n_estimators', 100),
                    max_depth=hyperparameters.get('max_depth', 6),
                    learning_rate=hyperparameters.get('learning_rate', 0.1),
                    random_state=hyperparameters.get('random_state', 42)
                )
            except ImportError:
                raise ModelTrainingError("XGBoost not available")
        else:
            raise ModelTrainingError(f"Unsupported model type: {model_type}")
    
    def _perform_cross_validation(self, model: Any, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any], 
                                 cv_folds: int) -> np.ndarray[Any, Any]:
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        
        self.logger.debug(
            "Cross-validation scores: %(scores)s (mean: %(mean).4f, std: %(std).4f)",
            source_module=self._source_module,
            context={
                "scores": cv_scores.tolist(),
                "mean": cv_scores.mean(),
                "std": cv_scores.std()
            }
        )
        
        return cv_scores
    
    def _calculate_performance_metrics(self, 
                                     model: Any,
                                     scaler: StandardScaler,
                                     X: pd.DataFrame,
                                     y: pd.Series[Any],
                                     symbol: str,
                                     start_time: datetime) -> ModelPerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        training_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ModelPerformanceMetrics(
            model_id=f"{symbol}_{model.__class__.__name__}",
            timestamp=datetime.utcnow(),
            mse=mse,
            mae=mae,
            r2_score=r2,
            accuracy_score=r2,  # Using R² as accuracy metric
            prediction_count=0,  # Will be updated during inference
            training_time_seconds=training_time,
            feature_count=len(X.columns),
            validation_score=r2
        )


class MLPredictionPipeline:
    """Enterprise-grade ML pipeline for price prediction."""
    
    def __init__(self, config: Dict[str, Any], logger: LoggerService) -> None:
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Pipeline components
        self.feature_engineer = FeatureEngineer(config.get('feature_engineering', {}), logger)
        self.model_trainer = ModelTrainer(config.get('model_training', {}), logger)
        
        # Model management
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.model_metrics: Dict[str, ModelPerformanceMetrics] = {}
        
        # Model storage paths
        self.model_storage_path = Path(config.get('model_storage_path', './models'))
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.prediction_stats = {
            'predictions_made': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'last_training_time': None,
            'last_prediction_time': None
        }
    
    async def train_and_deploy_model(self, 
                                   symbol: str, 
                                   training_data: pd.DataFrame,
                                   training_config: ModelTrainingConfig) -> str:
        """Complete training and deployment pipeline.
        
        Args:
            symbol: Trading symbol
            training_data: Raw OHLCV data
            training_config: Training configuration
            
        Returns:
            Model version identifier
            
        Raises:
            MLPipelineError: If training or deployment fails
        """
        try:
            self.logger.info(
                "Starting ML pipeline for %(symbol)s",
                source_module=self._source_module,
                context={"symbol": symbol}
            )
            
            # Feature engineering
            features_df = self.feature_engineer.engineer_features(training_data)
            
            # Model training
            training_result = await self.model_trainer.train_model(
                features_df, training_config, symbol
            )
            
            # Deploy model
            model_version = await self._deploy_model(symbol, training_result)
            
            # Update stats
            self.prediction_stats['last_training_time'] = datetime.utcnow()
            
            self.logger.info(
                "ML pipeline completed for %(symbol)s. Model version: %(version)s",
                source_module=self._source_module,
                context={"symbol": symbol, "version": model_version}
            )
            
            return model_version
            
        except Exception as e:
            self.logger.error(
                "ML pipeline failed for %(symbol)s: %(error)s",
                source_module=self._source_module,
                context={"symbol": symbol, "error": str(e)},
                exc_info=True
            )
            raise MLPipelineError(f"ML pipeline failed: {e}") from e
    
    async def predict_price(self, request: PredictionRequest) -> PredictionResult:
        """Generate price prediction with confidence intervals.
        
        Args:
            request: Prediction request
            
        Returns:
            Prediction result with confidence intervals
            
        Raises:
            MLPipelineError: If prediction fails
        """
        try:
            start_time = datetime.utcnow()
            symbol = request.symbol
            
            if symbol not in self.models:
                raise MLPipelineError(f"No trained model available for {symbol}")
            
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            feature_names = self.feature_names[symbol]
            
            # Prepare features
            feature_vector = self._prepare_prediction_features(request.features, feature_names)
            feature_vector_scaled = scaler.transform([feature_vector])
            
            # Make prediction
            predicted_price = model.predict(feature_vector_scaled)[0]
            
            # Calculate confidence interval
            prediction_uncertainty = self._estimate_prediction_uncertainty(
                model, feature_vector_scaled, request.confidence_level
            )
            
            confidence_interval = (
                predicted_price - prediction_uncertainty,
                predicted_price + prediction_uncertainty
            )
            
            # Get feature importance
            feature_importance = self._get_feature_importance(model, feature_names)
            
            # Update metrics
            if symbol in self.model_metrics:
                self.model_metrics[symbol].prediction_count += 1
            
            # Update stats
            self.prediction_stats['predictions_made'] += 1
            self.prediction_stats['successful_predictions'] += 1
            self.prediction_stats['last_prediction_time'] = datetime.utcnow()
            
            result = PredictionResult(
                symbol=symbol,
                predicted_price=float(predicted_price),
                confidence_interval=confidence_interval,
                prediction_timestamp=datetime.utcnow(),
                model_version=f"{symbol}_v1",  # Should be retrieved from model metadata
                feature_importance=feature_importance,
                request_id=request.request_id,
                model_id=f"{symbol}_model",
                accuracy_metrics=self._get_recent_accuracy_metrics(symbol)
            )
            
            prediction_time = (datetime.utcnow() - start_time).total_seconds()
            
            self.logger.debug(
                "Prediction completed for %(symbol)s: $%(price).2f (%(time).3fs)",
                source_module=self._source_module,
                context={
                    "symbol": symbol,
                    "price": predicted_price,
                    "time": prediction_time
                }
            )
            
            return result
            
        except Exception as e:
            self.prediction_stats['failed_predictions'] += 1
            self.logger.error(
                "Prediction failed for %(symbol)s: %(error)s",
                source_module=self._source_module,
                context={"symbol": request.symbol, "error": str(e)},
                exc_info=True
            )
            raise MLPipelineError(f"Prediction failed: {e}") from e
    
    async def _deploy_model(self, symbol: str, training_result: Dict[str, Any]) -> str:
        """Deploy trained model to production."""
        model = training_result["model"]
        scaler = training_result["scaler"]
        feature_names = training_result["feature_names"]
        metrics = training_result["metrics"]
        model_version = training_result["model_version"]
        
        # Store in memory
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        self.feature_names[symbol] = feature_names
        self.model_metrics[symbol] = metrics
        
        # Persist to disk
        model_path = self.model_storage_path / f"{symbol}_model.pkl"
        scaler_path = self.model_storage_path / f"{symbol}_scaler.pkl"
        metadata_path = self.model_storage_path / f"{symbol}_metadata.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        metadata = {
            "model_version": model_version,
            "feature_names": feature_names,
            "metrics": metrics,
            "deployment_time": datetime.utcnow(),
            "symbol": symbol
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        self.logger.info(
            "Model deployed for %(symbol)s at %(path)s",
            source_module=self._source_module,
            context={"symbol": symbol, "path": str(model_path)}
        )
        
        return model_version
    
    def _prepare_prediction_features(self, 
                                   features: Dict[str, float], 
                                   expected_features: List[str]) -> np.ndarray[Any, Any]:
        """Prepare feature vector for prediction."""
        feature_vector = []
        missing_features = []
        
        for feature_name in expected_features:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                feature_vector.append(0.0)  # Default value for missing features
                missing_features.append(feature_name)
        
        if missing_features:
            self.logger.warning(
                "Missing features for prediction: %(missing)s",
                source_module=self._source_module,
                context={"missing": missing_features}
            )
        
        return np.array(feature_vector, dtype=np.float32)
    
    def _estimate_prediction_uncertainty(self, 
                                       model: Any, 
                                       features: np.ndarray[Any, Any],
                                       confidence_level: float) -> float:
        """Estimate prediction uncertainty."""
        if hasattr(model, 'estimators_'):
            # For ensemble models, use prediction variance
            predictions = [estimator.predict(features)[0] for estimator in model.estimators_]
            std_dev = np.std(predictions)
            
            # Calculate confidence interval multiplier
            from scipy import stats
            confidence_multiplier = stats.norm.ppf((1 + confidence_level) / 2)
            
            return std_dev * confidence_multiplier
        else:
            # Default uncertainty estimation (could be improved with quantile regression)
            return 0.02  # 2% of price as default uncertainty
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from trained model."""
        feature_importance = {}
        
        if hasattr(model, 'feature_importances_'):
            for i, importance in enumerate(model.feature_importances_):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = float(importance)
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            for i, coef in enumerate(model.coef_):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = float(abs(coef))
        
        return feature_importance
    
    def _get_recent_accuracy_metrics(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get recent accuracy metrics for a symbol."""
        if symbol not in self.model_metrics:
            return None
        
        metrics = self.model_metrics[symbol]
        return {
            "mse": metrics.mse,
            "mae": metrics.mae,
            "r2_score": metrics.r2_score,
            "validation_score": metrics.validation_score
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        return {
            "models_deployed": len(self.models),
            "prediction_stats": self.prediction_stats.copy(),
            "model_symbols": list[Any](self.models.keys()),
            "storage_path": str(self.model_storage_path),
            "feature_engineering_config": self.config.get('feature_engineering', {}),
            "last_training": self.prediction_stats.get('last_training_time'),
            "last_prediction": self.prediction_stats.get('last_prediction_time')
        }
    
    def get_model_performance(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed model performance metrics."""
        if symbol not in self.model_metrics:
            return None
        
        metrics = self.model_metrics[symbol]
        return {
            "symbol": symbol,
            "model_id": metrics.model_id,
            "timestamp": metrics.timestamp,
            "performance_metrics": {
                "mse": metrics.mse,
                "mae": metrics.mae,
                "r2_score": metrics.r2_score,
                "validation_score": metrics.validation_score
            },
            "operational_metrics": {
                "prediction_count": metrics.prediction_count,
                "training_time_seconds": metrics.training_time_seconds,
                "feature_count": metrics.feature_count
            }
        } 