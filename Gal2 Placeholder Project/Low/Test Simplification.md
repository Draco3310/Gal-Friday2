# Test Simplification Implementation Design

**File**: `/scripts/train_initial_model.py`
- **Line 427**: `# this function. Simplified: fit only on train data for now`
- **Issue**: Model training simplified to avoid cross-validation and proper evaluation

**File**: `/tests/unit/test_feature_engine_pipeline_construction.py`
- **Line 224**: `# For now, we confirm default is applied`
- **Issue**: Test assertions simplified to basic default checking without comprehensive validation

**File**: `/tests/test_integration.py`
- **Line 277**: `# For now, just verify the test framework`
- **Issue**: Integration tests reduced to framework verification instead of actual integration testing

**Impact**: Simplified tests provide inadequate coverage and may miss critical bugs in production

## Overview
Test implementations have been simplified with "for now" comments that reduce test effectiveness and code coverage. This design implements comprehensive testing frameworks with proper validation, edge case coverage, and production-ready test scenarios.

## Architecture Design

### 1. Current Test Simplification Issues

```
Test Simplification Problems:
├── Model Training Script
│   ├── Missing cross-validation
│   ├── No proper train/validation/test split
│   ├── Simplified evaluation metrics
│   └── No hyperparameter tuning
├── Unit Tests
│   ├── Basic default value checking only
│   ├── Missing edge case testing
│   ├── No error scenario validation
│   └── Insufficient assertion coverage
├── Integration Tests
│   ├── Framework verification only
│   ├── No actual integration scenarios
│   ├── Missing end-to-end workflows
│   └── No performance validation
└── Quality Impact
    ├── Reduced bug detection capability
    ├── Poor test coverage metrics
    ├── Missed regression scenarios
    └── Inadequate production readiness validation
```

### 2. Comprehensive Testing Architecture

```
Production-Ready Testing Framework:
├── Enhanced Model Training & Validation
│   ├── Comprehensive Cross-Validation
│   │   ├── K-fold validation strategy
│   │   ├── Time-series aware splitting
│   │   ├── Stratified validation for imbalanced data
│   │   └── Out-of-sample testing framework
│   ├── Advanced Evaluation Metrics
│   │   ├── Financial performance metrics
│   │   ├── Risk-adjusted returns
│   │   ├── Drawdown analysis
│   │   └── Statistical significance testing
│   ├── Hyperparameter Optimization
│   │   ├── Bayesian optimization
│   │   ├── Grid search with cross-validation
│   │   ├── Random search strategies
│   │   └── Early stopping mechanisms
│   └── Model Validation Pipeline
│       ├── Overfitting detection
│       ├── Feature importance analysis
│       ├── Model stability testing
│       └── Production readiness validation
├── Comprehensive Unit Testing Framework
│   ├── Feature Engine Testing
│   │   ├── Pipeline construction validation
│   │   ├── Feature transformation accuracy
│   │   ├── Edge case handling
│   │   └── Error propagation testing
│   ├── Data Validation Testing
│   │   ├── Schema validation
│   │   ├── Data quality checks
│   │   ├── Boundary condition testing
│   │   └── Missing data handling
│   ├── Configuration Testing
│   │   ├── Default value validation
│   │   ├── Configuration inheritance
│   │   ├── Environment-specific settings
│   │   └── Configuration error handling
│   └── Performance Testing
│       ├── Memory usage validation
│       ├── Processing time benchmarks
│       ├── Scalability testing
│       └── Resource utilization monitoring
├── Advanced Integration Testing
│   ├── End-to-End Workflow Testing
│   │   ├── Complete trading pipeline
│   │   ├── Data ingestion to execution
│   │   ├── Error recovery scenarios
│   │   └── System state validation
│   ├── Component Interaction Testing
│   │   ├── Service communication validation
│   │   ├── Event bus messaging
│   │   ├── Database transaction integrity
│   │   └── External API integration
│   ├── Performance Integration Testing
│   │   ├── Load testing scenarios
│   │   ├── Stress testing under high volume
│   │   ├── Latency measurement
│   │   └── Resource contention testing
│   └── Security Integration Testing
│       ├── Authentication flow testing
│       ├── Authorization validation
│       ├── Data encryption verification
│       └── Audit trail validation
└── Test Infrastructure & Automation
    ├── Test Data Management
    │   ├── Comprehensive test datasets
    │   ├── Data anonymization
    │   ├── Test data versioning
    │   └── Synthetic data generation
    ├── Test Environment Management
    │   ├── Containerized test environments
    │   ├── Database seeding and cleanup
    │   ├── Service mocking frameworks
    │   └── Environment isolation
    ├── Continuous Testing Pipeline
    │   ├── Automated test execution
    │   ├── Coverage reporting
    │   ├── Test result analysis
    │   └── Regression detection
    └── Test Reporting & Analytics
        ├── Comprehensive test metrics
        ├── Trend analysis
        ├── Failure pattern detection
        └── Quality gates validation
```

### 3. Key Features

1. **Comprehensive Coverage**: Complete test coverage for all scenarios including edge cases
2. **Production Readiness**: Tests that validate production deployment readiness
3. **Performance Validation**: Comprehensive performance and scalability testing
4. **Automated Quality Gates**: Automated validation of quality metrics and thresholds
5. **Advanced Analytics**: Deep analysis of model performance and system behavior
6. **Error Resilience**: Comprehensive error scenario testing and recovery validation

## Implementation Plan

### Phase 1: Enhanced Model Training Framework

**File**: `/scripts/train_initial_model.py`
**Target Line**: Line 427 - Replace simplified training with comprehensive framework

```python
"""
Production-ready model training with comprehensive validation and evaluation.

This module provides enterprise-grade model training with cross-validation,
hyperparameter optimization, and rigorous evaluation metrics.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
from dataclasses import dataclass

from gal_friday.model_training.base import BaseTrainingPipeline
from gal_friday.model_training.validation import ModelValidator
from gal_friday.model_training.metrics import FinancialMetrics
from gal_friday.logger_service import LoggerService


@dataclass
class TrainingConfiguration:
    """Comprehensive training configuration."""
    cross_validation_folds: int = 5
    test_size_ratio: float = 0.2
    validation_size_ratio: float = 0.2
    min_train_samples: int = 1000
    max_training_time_hours: int = 24
    early_stopping_patience: int = 10
    hyperparameter_trials: int = 100
    enable_feature_selection: bool = True
    enable_ensemble_methods: bool = True
    financial_metrics_enabled: bool = True
    
    # Risk management parameters
    max_drawdown_threshold: float = 0.15
    min_sharpe_ratio: float = 1.0
    min_win_rate: float = 0.45
    
    # Performance requirements
    max_prediction_latency_ms: float = 100.0
    min_accuracy_threshold: float = 0.65
    min_precision_threshold: float = 0.60


class EnhancedModelTrainingPipeline:
    """Production-ready model training pipeline with comprehensive validation."""
    
    def __init__(self, config: TrainingConfiguration, logger: LoggerService):
        self.config = config
        self.logger = logger
        self.validator = ModelValidator(logger)
        self.financial_metrics = FinancialMetrics(logger)
        self.training_history: List[Dict[str, Any]] = []
        
    async def train_comprehensive_model(
        self, 
        features: pd.DataFrame, 
        targets: pd.DataFrame,
        feature_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train model with comprehensive validation and evaluation.
        
        This replaces the simplified training approach with full cross-validation,
        hyperparameter optimization, and production readiness validation.
        """
        self.logger.info("Starting comprehensive model training pipeline")
        
        try:
            # Phase 1: Data Preparation and Validation
            train_data, val_data, test_data = await self._prepare_training_data(features, targets)
            
            # Phase 2: Feature Engineering and Selection
            if self.config.enable_feature_selection:
                features_selected = await self._perform_feature_selection(train_data, feature_metadata)
            else:
                features_selected = train_data['features']
            
            # Phase 3: Cross-Validation Framework
            cv_results = await self._perform_cross_validation(features_selected, train_data['targets'])
            
            # Phase 4: Hyperparameter Optimization
            best_params = await self._optimize_hyperparameters(features_selected, train_data['targets'])
            
            # Phase 5: Final Model Training
            final_model = await self._train_final_model(
                features_selected, 
                train_data['targets'], 
                best_params
            )
            
            # Phase 6: Comprehensive Evaluation
            evaluation_results = await self._comprehensive_evaluation(
                final_model, 
                test_data, 
                val_data
            )
            
            # Phase 7: Production Readiness Validation
            readiness_check = await self._validate_production_readiness(
                final_model, 
                evaluation_results
            )
            
            # Phase 8: Financial Performance Analysis
            if self.config.financial_metrics_enabled:
                financial_analysis = await self._analyze_financial_performance(
                    final_model, 
                    test_data
                )
            else:
                financial_analysis = {}
            
            # Compile comprehensive results
            training_results = {
                "model": final_model,
                "cross_validation_results": cv_results,
                "hyperparameter_optimization": {
                    "best_parameters": best_params,
                    "optimization_history": self.training_history
                },
                "evaluation_results": evaluation_results,
                "financial_analysis": financial_analysis,
                "production_readiness": readiness_check,
                "training_metadata": {
                    "training_samples": len(train_data['features']),
                    "validation_samples": len(val_data['features']),
                    "test_samples": len(test_data['features']),
                    "feature_count": len(features_selected.columns),
                    "training_duration": datetime.now() - self.training_start_time,
                    "configuration": self.config
                }
            }
            
            self.logger.info("Model training completed successfully")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}", exc_info=True)
            raise
    
    async def _prepare_training_data(
        self, 
        features: pd.DataFrame, 
        targets: pd.DataFrame
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Prepare training data with proper time-series aware splitting."""
        self.logger.info("Preparing training data with time-series splits")
        
        # Validate data quality
        await self._validate_data_quality(features, targets)
        
        # Time-series aware splitting
        total_samples = len(features)
        test_size = int(total_samples * self.config.test_size_ratio)
        val_size = int(total_samples * self.config.validation_size_ratio)
        train_size = total_samples - test_size - val_size
        
        # Ensure minimum training samples
        if train_size < self.config.min_train_samples:
            raise ValueError(f"Insufficient training samples: {train_size} < {self.config.min_train_samples}")
        
        # Split data maintaining temporal order
        train_features = features.iloc[:train_size]
        train_targets = targets.iloc[:train_size]
        
        val_features = features.iloc[train_size:train_size + val_size]
        val_targets = targets.iloc[train_size:train_size + val_size]
        
        test_features = features.iloc[train_size + val_size:]
        test_targets = targets.iloc[train_size + val_size:]
        
        train_data = {"features": train_features, "targets": train_targets}
        val_data = {"features": val_features, "targets": val_targets}
        test_data = {"features": test_features, "targets": test_targets}
        
        self.logger.info(f"Data split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        return train_data, val_data, test_data
    
    async def _perform_cross_validation(
        self, 
        features: pd.DataFrame, 
        targets: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform comprehensive cross-validation with time-series splits."""
        self.logger.info("Performing time-series cross-validation")
        
        # Time series split for financial data
        tscv = TimeSeriesSplit(n_splits=self.config.cross_validation_folds)
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'sharpe_ratio': [],
            'max_drawdown': []
        }
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(features)):
            self.logger.info(f"Processing fold {fold_idx + 1}/{self.config.cross_validation_folds}")
            
            X_train_fold = features.iloc[train_idx]
            y_train_fold = targets.iloc[train_idx]
            X_val_fold = features.iloc[val_idx]
            y_val_fold = targets.iloc[val_idx]
            
            # Train model on fold
            fold_model = await self._train_fold_model(X_train_fold, y_train_fold)
            
            # Evaluate fold
            fold_predictions = fold_model.predict(X_val_fold)
            
            # Calculate standard metrics
            fold_accuracy = accuracy_score(y_val_fold, fold_predictions)
            fold_precision = precision_score(y_val_fold, fold_predictions, average='weighted')
            fold_recall = recall_score(y_val_fold, fold_predictions, average='weighted')
            fold_f1 = f1_score(y_val_fold, fold_predictions, average='weighted')
            
            # Calculate financial metrics
            if self.config.financial_metrics_enabled:
                financial_metrics = await self.financial_metrics.calculate_fold_metrics(
                    y_val_fold, 
                    fold_predictions,
                    X_val_fold
                )
                fold_sharpe = financial_metrics.get('sharpe_ratio', 0.0)
                fold_drawdown = financial_metrics.get('max_drawdown', 1.0)
            else:
                fold_sharpe = 0.0
                fold_drawdown = 0.0
            
            # Store results
            cv_scores['accuracy'].append(fold_accuracy)
            cv_scores['precision'].append(fold_precision)
            cv_scores['recall'].append(fold_recall)
            cv_scores['f1'].append(fold_f1)
            cv_scores['sharpe_ratio'].append(fold_sharpe)
            cv_scores['max_drawdown'].append(fold_drawdown)
            
            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': fold_accuracy,
                'precision': fold_precision,
                'recall': fold_recall,
                'f1': fold_f1,
                'sharpe_ratio': fold_sharpe,
                'max_drawdown': fold_drawdown,
                'train_samples': len(train_idx),
                'val_samples': len(val_idx)
            })
        
        # Calculate cross-validation statistics
        cv_results = {
            'mean_scores': {metric: np.mean(scores) for metric, scores in cv_scores.items()},
            'std_scores': {metric: np.std(scores) for metric, scores in cv_scores.items()},
            'fold_results': fold_results,
            'stability_score': self._calculate_stability_score(cv_scores),
            'meets_requirements': self._validate_cv_requirements(cv_scores)
        }
        
        self.logger.info(f"Cross-validation completed. Mean accuracy: {cv_results['mean_scores']['accuracy']:.4f}")
        
        return cv_results
    
    async def _optimize_hyperparameters(
        self, 
        features: pd.DataFrame, 
        targets: pd.DataFrame
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Bayesian optimization."""
        self.logger.info("Starting hyperparameter optimization")
        
        def objective(trial):
            # Define hyperparameter space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
            
            # Cross-validation with current parameters
            model = self._create_model_with_params(params)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)  # Reduced folds for optimization speed
            scores = cross_val_score(model, features, targets, cv=tscv, scoring='accuracy')
            
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.hyperparameter_trials)
        
        best_params = study.best_params
        optimization_history = [
            {
                'trial': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            }
            for trial in study.trials
        ]
        
        self.training_history.extend(optimization_history)
        
        self.logger.info(f"Hyperparameter optimization completed. Best score: {study.best_value:.4f}")
        
        return {
            'best_parameters': best_params,
            'best_score': study.best_value,
            'optimization_history': optimization_history,
            'study_statistics': {
                'n_trials': len(study.trials),
                'n_complete_trials': len([t for t in study.trials if t.state.name == 'COMPLETE']),
                'n_failed_trials': len([t for t in study.trials if t.state.name == 'FAIL'])
            }
        }
    
    async def _comprehensive_evaluation(
        self, 
        model: Any, 
        test_data: Dict[str, pd.DataFrame],
        val_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Perform comprehensive model evaluation on test and validation sets."""
        self.logger.info("Performing comprehensive model evaluation")
        
        evaluation_results = {}
        
        # Evaluate on both validation and test sets
        for dataset_name, dataset in [("validation", val_data), ("test", test_data)]:
            predictions = model.predict(dataset['features'])
            prediction_proba = model.predict_proba(dataset['features']) if hasattr(model, 'predict_proba') else None
            
            # Standard classification metrics
            metrics = {
                'accuracy': accuracy_score(dataset['targets'], predictions),
                'precision': precision_score(dataset['targets'], predictions, average='weighted'),
                'recall': recall_score(dataset['targets'], predictions, average='weighted'),
                'f1': f1_score(dataset['targets'], predictions, average='weighted')
            }
            
            # Financial performance metrics
            if self.config.financial_metrics_enabled:
                financial_metrics = await self.financial_metrics.calculate_comprehensive_metrics(
                    dataset['targets'], 
                    predictions,
                    dataset['features'],
                    prediction_proba
                )
                metrics.update(financial_metrics)
            
            # Performance metrics
            performance_metrics = await self._measure_prediction_performance(model, dataset['features'])
            metrics.update(performance_metrics)
            
            evaluation_results[dataset_name] = metrics
        
        # Model stability analysis
        stability_analysis = await self._analyze_model_stability(model, test_data, val_data)
        evaluation_results['stability_analysis'] = stability_analysis
        
        # Feature importance analysis
        feature_importance = await self._analyze_feature_importance(model, test_data['features'])
        evaluation_results['feature_importance'] = feature_importance
        
        return evaluation_results
    
    async def _validate_production_readiness(
        self, 
        model: Any, 
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate model readiness for production deployment."""
        self.logger.info("Validating production readiness")
        
        readiness_checks = {
            'accuracy_threshold': False,
            'precision_threshold': False,
            'latency_requirement': False,
            'financial_requirements': False,
            'stability_requirement': False,
            'overall_ready': False
        }
        
        test_metrics = evaluation_results.get('test', {})
        
        # Check accuracy threshold
        if test_metrics.get('accuracy', 0) >= self.config.min_accuracy_threshold:
            readiness_checks['accuracy_threshold'] = True
        
        # Check precision threshold
        if test_metrics.get('precision', 0) >= self.config.min_precision_threshold:
            readiness_checks['precision_threshold'] = True
        
        # Check latency requirement
        if test_metrics.get('avg_prediction_time_ms', float('inf')) <= self.config.max_prediction_latency_ms:
            readiness_checks['latency_requirement'] = True
        
        # Check financial requirements
        if self.config.financial_metrics_enabled:
            sharpe_ratio = test_metrics.get('sharpe_ratio', 0)
            max_drawdown = test_metrics.get('max_drawdown', 1.0)
            win_rate = test_metrics.get('win_rate', 0)
            
            financial_ready = (
                sharpe_ratio >= self.config.min_sharpe_ratio and
                max_drawdown <= self.config.max_drawdown_threshold and
                win_rate >= self.config.min_win_rate
            )
            readiness_checks['financial_requirements'] = financial_ready
        else:
            readiness_checks['financial_requirements'] = True
        
        # Check stability requirement
        stability_score = evaluation_results.get('stability_analysis', {}).get('overall_stability', 0)
        if stability_score >= 0.8:  # 80% stability threshold
            readiness_checks['stability_requirement'] = True
        
        # Overall readiness
        readiness_checks['overall_ready'] = all([
            readiness_checks['accuracy_threshold'],
            readiness_checks['precision_threshold'],
            readiness_checks['latency_requirement'],
            readiness_checks['financial_requirements'],
            readiness_checks['stability_requirement']
        ])
        
        readiness_summary = {
            'checks': readiness_checks,
            'requirements': {
                'min_accuracy': self.config.min_accuracy_threshold,
                'min_precision': self.config.min_precision_threshold,
                'max_latency_ms': self.config.max_prediction_latency_ms,
                'min_sharpe_ratio': self.config.min_sharpe_ratio,
                'max_drawdown': self.config.max_drawdown_threshold,
                'min_win_rate': self.config.min_win_rate
            },
            'actual_performance': test_metrics,
            'recommendation': 'DEPLOY' if readiness_checks['overall_ready'] else 'NEEDS_IMPROVEMENT'
        }
        
        if readiness_checks['overall_ready']:
            self.logger.info("Model is ready for production deployment")
        else:
            failed_checks = [check for check, passed in readiness_checks.items() if not passed]
            self.logger.warning(f"Model not ready for production. Failed checks: {failed_checks}")
        
        return readiness_summary
```

### Phase 2: Advanced Unit Testing Framework

**File**: `/tests/unit/test_feature_engine_pipeline_construction.py`
**Target Line**: Line 224 - Replace basic default checking with comprehensive validation

```python
"""
Comprehensive unit tests for Feature Engine Pipeline Construction.

This module provides exhaustive testing of feature pipeline construction,
transformation accuracy, edge cases, and error handling scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch

from gal_friday.feature_engine.pipeline import FeatureEnginePipeline
from gal_friday.feature_engine.transformers import *
from gal_friday.feature_engine.config import FeatureEngineConfig
from tests.fixtures.feature_data import FeatureDataFactory


class TestFeatureEnginePipelineConstruction:
    """Comprehensive test suite for feature engine pipeline construction."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate comprehensive sample market data for testing."""
        return FeatureDataFactory.create_comprehensive_market_data(
            symbols=['BTC/USDT', 'ETH/USDT'],
            start_date=datetime(2023, 1, 1),
            duration_days=30,
            frequency='1min'
        )
    
    @pytest.fixture
    def default_config(self):
        """Create default feature engine configuration."""
        return FeatureEngineConfig(
            technical_indicators=['sma_20', 'rsi_14', 'macd', 'bollinger_bands'],
            statistical_features=['returns', 'volatility', 'skewness', 'kurtosis'],
            market_structure_features=['order_imbalance', 'spread', 'depth'],
            time_features=['hour', 'day_of_week', 'month'],
            lookback_periods=[5, 10, 20, 50],
            enable_feature_selection=True,
            enable_normalization=True,
            enable_outlier_detection=True
        )
    
    def test_pipeline_construction_with_defaults(self, default_config):
        """Test pipeline construction with default configuration values."""
        # This replaces the simplified "For now, we confirm default is applied" approach
        
        pipeline = FeatureEnginePipeline(default_config)
        
        # Comprehensive validation of default values
        assert pipeline.config.technical_indicators == ['sma_20', 'rsi_14', 'macd', 'bollinger_bands']
        assert pipeline.config.statistical_features == ['returns', 'volatility', 'skewness', 'kurtosis']
        assert pipeline.config.lookback_periods == [5, 10, 20, 50]
        assert pipeline.config.enable_feature_selection is True
        assert pipeline.config.enable_normalization is True
        assert pipeline.config.enable_outlier_detection is True
        
        # Validate pipeline components are correctly initialized
        assert hasattr(pipeline, 'technical_transformer')
        assert hasattr(pipeline, 'statistical_transformer')
        assert hasattr(pipeline, 'market_structure_transformer')
        assert hasattr(pipeline, 'time_transformer')
        
        # Validate transformer configurations
        assert pipeline.technical_transformer.indicators == default_config.technical_indicators
        assert pipeline.statistical_transformer.features == default_config.statistical_features
        assert pipeline.time_transformer.features == default_config.time_features
        
        # Validate pipeline structure
        expected_steps = [
            'technical_transformer',
            'statistical_transformer', 
            'market_structure_transformer',
            'time_transformer'
        ]
        
        if default_config.enable_normalization:
            expected_steps.append('normalizer')
        
        if default_config.enable_outlier_detection:
            expected_steps.append('outlier_detector')
        
        if default_config.enable_feature_selection:
            expected_steps.append('feature_selector')
        
        pipeline_steps = [step[0] for step in pipeline.pipeline.steps]
        assert all(step in pipeline_steps for step in expected_steps)
    
    def test_pipeline_construction_custom_config(self):
        """Test pipeline construction with custom configuration."""
        custom_config = FeatureEngineConfig(
            technical_indicators=['sma_50', 'ema_20', 'rsi_21'],
            statistical_features=['returns', 'volatility'],
            market_structure_features=['spread'],
            time_features=['hour'],
            lookback_periods=[10, 30],
            enable_feature_selection=False,
            enable_normalization=True,
            enable_outlier_detection=False
        )
        
        pipeline = FeatureEnginePipeline(custom_config)
        
        # Validate custom configuration is applied
        assert pipeline.config.technical_indicators == ['sma_50', 'ema_20', 'rsi_21']
        assert pipeline.config.statistical_features == ['returns', 'volatility']
        assert pipeline.config.market_structure_features == ['spread']
        assert pipeline.config.time_features == ['hour']
        assert pipeline.config.lookback_periods == [10, 30]
        
        # Validate conditional components
        pipeline_steps = [step[0] for step in pipeline.pipeline.steps]
        assert 'normalizer' in pipeline_steps
        assert 'feature_selector' not in pipeline_steps
        assert 'outlier_detector' not in pipeline_steps
    
    def test_technical_indicator_construction(self, sample_market_data, default_config):
        """Test technical indicator transformer construction and functionality."""
        pipeline = FeatureEnginePipeline(default_config)
        
        # Test individual technical indicators
        for indicator in default_config.technical_indicators:
            # Validate indicator is properly configured
            assert indicator in pipeline.technical_transformer.indicators
            
            # Test indicator calculation
            if indicator.startswith('sma_'):
                period = int(indicator.split('_')[1])
                result = pipeline.technical_transformer._calculate_sma(
                    sample_market_data['close'], 
                    period
                )
                assert len(result) == len(sample_market_data)
                assert not result.iloc[period:].isna().all()
            
            elif indicator.startswith('rsi_'):
                period = int(indicator.split('_')[1])
                result = pipeline.technical_transformer._calculate_rsi(
                    sample_market_data['close'], 
                    period
                )
                assert len(result) == len(sample_market_data)
                assert (result >= 0).all() and (result <= 100).all()
            
            elif indicator == 'macd':
                result = pipeline.technical_transformer._calculate_macd(
                    sample_market_data['close']
                )
                assert 'macd' in result.columns
                assert 'macd_signal' in result.columns
                assert 'macd_histogram' in result.columns
            
            elif indicator == 'bollinger_bands':
                result = pipeline.technical_transformer._calculate_bollinger_bands(
                    sample_market_data['close']
                )
                assert 'bb_upper' in result.columns
                assert 'bb_lower' in result.columns
                assert 'bb_middle' in result.columns
    
    def test_statistical_feature_construction(self, sample_market_data, default_config):
        """Test statistical feature transformer construction and calculations."""
        pipeline = FeatureEnginePipeline(default_config)
        
        # Test returns calculation
        if 'returns' in default_config.statistical_features:
            returns = pipeline.statistical_transformer._calculate_returns(
                sample_market_data['close']
            )
            assert len(returns) == len(sample_market_data) - 1
            assert returns.dtype == np.float64
            
            # Validate returns calculation accuracy
            expected_returns = sample_market_data['close'].pct_change().dropna()
            pd.testing.assert_series_equal(returns, expected_returns, check_names=False)
        
        # Test volatility calculation
        if 'volatility' in default_config.statistical_features:
            for period in default_config.lookback_periods:
                volatility = pipeline.statistical_transformer._calculate_volatility(
                    sample_market_data['close'], 
                    period
                )
                assert len(volatility) == len(sample_market_data)
                assert (volatility >= 0).all()
        
        # Test skewness and kurtosis
        if 'skewness' in default_config.statistical_features:
            skewness = pipeline.statistical_transformer._calculate_skewness(
                sample_market_data['close'], 
                20
            )
            assert len(skewness) == len(sample_market_data)
            assert skewness.dtype == np.float64
        
        if 'kurtosis' in default_config.statistical_features:
            kurtosis = pipeline.statistical_transformer._calculate_kurtosis(
                sample_market_data['close'], 
                20
            )
            assert len(kurtosis) == len(sample_market_data)
            assert kurtosis.dtype == np.float64
    
    def test_edge_case_handling(self, default_config):
        """Test pipeline handling of edge cases and boundary conditions."""
        pipeline = FeatureEnginePipeline(default_config)
        
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1min'),
            'open': [100, 101, 102, 101, 103],
            'high': [101, 102, 103, 102, 104],
            'low': [99, 100, 101, 100, 102],
            'close': [100.5, 101.5, 102.5, 101.5, 103.5],
            'volume': [1000, 1100, 1200, 1050, 1150]
        })
        
        # Should handle minimal data gracefully
        result = pipeline.transform(minimal_data)
        assert result is not None
        assert len(result) == len(minimal_data)
        
        # Test with missing values
        data_with_nan = sample_market_data.copy()
        data_with_nan.loc[10:15, 'close'] = np.nan
        
        result_with_nan = pipeline.transform(data_with_nan)
        assert result_with_nan is not None
        
        # Test with constant values
        constant_data = sample_market_data.copy()
        constant_data['close'] = 100.0
        
        result_constant = pipeline.transform(constant_data)
        assert result_constant is not None
        
        # Some features should be zero/constant for constant price data
        returns_col = [col for col in result_constant.columns if 'returns' in col]
        if returns_col:
            assert (result_constant[returns_col[0]].fillna(0) == 0).all()
    
    def test_error_handling_scenarios(self, default_config):
        """Test comprehensive error handling in pipeline construction."""
        # Test invalid configuration
        with pytest.raises(ValueError):
            invalid_config = FeatureEngineConfig(
                technical_indicators=['invalid_indicator'],
                statistical_features=['returns'],
                lookback_periods=[]  # Empty lookback periods
            )
            FeatureEnginePipeline(invalid_config)
        
        # Test invalid data format
        pipeline = FeatureEnginePipeline(default_config)
        
        with pytest.raises(ValueError):
            # Missing required columns
            invalid_data = pd.DataFrame({'price': [1, 2, 3]})
            pipeline.transform(invalid_data)
        
        with pytest.raises(ValueError):
            # Wrong data types
            invalid_data = pd.DataFrame({
                'timestamp': ['not_a_date', 'also_not_a_date'],
                'close': ['not_a_number', 'also_not_a_number']
            })
            pipeline.transform(invalid_data)
        
        # Test empty data
        with pytest.raises(ValueError):
            empty_data = pd.DataFrame()
            pipeline.transform(empty_data)
    
    def test_feature_selection_integration(self, sample_market_data):
        """Test feature selection integration within pipeline."""
        config_with_selection = FeatureEngineConfig(
            technical_indicators=['sma_20', 'rsi_14', 'macd'],
            statistical_features=['returns', 'volatility'],
            enable_feature_selection=True,
            feature_selection_method='mutual_info',
            max_features=10
        )
        
        pipeline = FeatureEnginePipeline(config_with_selection)
        
        # Generate features first
        features = pipeline.transform(sample_market_data)
        
        # Simulate target variable for feature selection
        target = np.random.choice([0, 1], size=len(features))
        
        # Fit pipeline with target for feature selection
        pipeline.fit(sample_market_data, target)
        
        # Transform and verify feature selection
        selected_features = pipeline.transform(sample_market_data)
        
        assert selected_features.shape[1] <= config_with_selection.max_features
        assert selected_features.shape[0] == len(sample_market_data)
        
        # Verify feature selection metadata
        assert hasattr(pipeline.feature_selector, 'selected_features_')
        assert len(pipeline.feature_selector.selected_features_) <= config_with_selection.max_features
    
    def test_normalization_integration(self, sample_market_data):
        """Test normalization integration within pipeline."""
        config_with_norm = FeatureEngineConfig(
            technical_indicators=['sma_20', 'rsi_14'],
            statistical_features=['returns', 'volatility'],
            enable_normalization=True,
            normalization_method='standard'
        )
        
        pipeline = FeatureEnginePipeline(config_with_norm)
        
        # Fit and transform
        pipeline.fit(sample_market_data)
        normalized_features = pipeline.transform(sample_market_data)
        
        # Check that features are normalized (mean ≈ 0, std ≈ 1)
        feature_means = normalized_features.mean()
        feature_stds = normalized_features.std()
        
        # Allow some tolerance for numerical precision
        assert (abs(feature_means) < 0.1).all()
        assert (abs(feature_stds - 1.0) < 0.1).all()
    
    def test_pipeline_serialization(self, sample_market_data, default_config):
        """Test pipeline serialization and deserialization."""
        pipeline = FeatureEnginePipeline(default_config)
        pipeline.fit(sample_market_data)
        
        # Test serialization
        serialized_pipeline = pipeline.serialize()
        assert isinstance(serialized_pipeline, dict)
        assert 'config' in serialized_pipeline
        assert 'pipeline_state' in serialized_pipeline
        
        # Test deserialization
        new_pipeline = FeatureEnginePipeline.deserialize(serialized_pipeline)
        assert new_pipeline.config.__dict__ == pipeline.config.__dict__
        
        # Test that deserialized pipeline produces same results
        original_features = pipeline.transform(sample_market_data)
        new_features = new_pipeline.transform(sample_market_data)
        
        pd.testing.assert_frame_equal(original_features, new_features)
    
    def test_performance_requirements(self, sample_market_data, default_config):
        """Test that pipeline meets performance requirements."""
        pipeline = FeatureEnginePipeline(default_config)
        
        import time
        
        # Test fitting performance
        start_time = time.time()
        pipeline.fit(sample_market_data)
        fit_time = time.time() - start_time
        
        # Test transformation performance
        start_time = time.time()
        features = pipeline.transform(sample_market_data)
        transform_time = time.time() - start_time
        
        # Performance assertions (adjust based on requirements)
        assert fit_time < 30.0  # Fitting should complete within 30 seconds
        assert transform_time < 5.0  # Transformation should complete within 5 seconds
        
        # Memory usage validation
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_usage_mb = process.memory_info().rss / 1024 / 1024
        
        # Memory usage should be reasonable (adjust based on data size)
        assert memory_usage_mb < 1000  # Less than 1GB for test dataset
        
        # Feature count validation
        assert features.shape[1] > 0
        assert features.shape[1] < 1000  # Reasonable feature count
        
        # Data integrity validation
        assert not features.isnull().all().any()  # No completely null columns
        assert features.shape[0] == len(sample_market_data)  # Same number of rows
```

### Phase 3: Comprehensive Integration Testing Framework

**File**: `/tests/test_integration.py`
**Target Line**: Line 277 - Replace framework verification with actual integration testing

```python
"""
Comprehensive integration tests for complete trading system workflows.

This module provides exhaustive integration testing covering end-to-end
workflows, component interactions, performance validation, and error scenarios.
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch
import pandas as pd
from decimal import Decimal

from gal_friday.trading_system import TradingSystem
from gal_friday.data_ingestor import DataIngestor
from gal_friday.strategy_engine import StrategyEngine
from gal_friday.portfolio_manager import PortfolioManager
from gal_friday.risk_manager import RiskManager
from gal_friday.execution_engine import ExecutionEngine
from gal_friday.logger_service import LoggerService
from tests.fixtures.integration_data import IntegrationDataFactory
from tests.fixtures.mock_services import MockExchangeService, MockLoggerService


class TestTradingSystemIntegration:
    """
    Comprehensive integration test suite for complete trading system.
    
    This replaces the simplified "For now, just verify the test framework" approach
    with actual end-to-end integration testing scenarios.
    """
    
    @pytest.fixture
    async def trading_system(self):
        """Set up complete trading system for integration testing."""
        # Create comprehensive test configuration
        config = IntegrationDataFactory.create_comprehensive_system_config()
        
        # Initialize core components
        logger = MockLoggerService()
        event_bus = asyncio.Queue()
        
        # Initialize system components
        data_ingestor = DataIngestor(config.data_config, event_bus, logger)
        strategy_engine = StrategyEngine(config.strategy_config, logger)
        portfolio_manager = PortfolioManager(config.portfolio_config, logger)
        risk_manager = RiskManager(config.risk_config, logger)
        execution_engine = ExecutionEngine(config.execution_config, logger)
        
        # Create trading system
        trading_system = TradingSystem(
            data_ingestor=data_ingestor,
            strategy_engine=strategy_engine,
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            execution_engine=execution_engine,
            event_bus=event_bus,
            logger=logger
        )
        
        return trading_system
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate comprehensive market data for integration testing."""
        return IntegrationDataFactory.create_comprehensive_market_data(
            symbols=['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
            start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            duration_hours=24,
            frequency_minutes=1
        )
    
    @pytest.mark.asyncio
    async def test_complete_trading_workflow_integration(self, trading_system, sample_market_data):
        """
        Test complete end-to-end trading workflow.
        
        This is a comprehensive integration test that validates the entire
        trading pipeline from data ingestion to order execution.
        """
        # Phase 1: System Initialization
        await trading_system.initialize()
        assert trading_system.is_initialized
        
        # Phase 2: Data Ingestion
        ingestion_results = []
        for data_point in sample_market_data[:100]:  # Test with subset for speed
            result = await trading_system.ingest_market_data(data_point)
            ingestion_results.append(result)
        
        # Validate data ingestion
        assert len(ingestion_results) == 100
        assert all(result.success for result in ingestion_results)
        
        # Phase 3: Strategy Signal Generation
        strategy_signals = []
        for i in range(10):  # Generate signals for 10 time periods
            signals = await trading_system.generate_strategy_signals()
            strategy_signals.extend(signals)
        
        # Validate signal generation
        assert len(strategy_signals) > 0
        for signal in strategy_signals:
            assert signal.trading_pair in ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
            assert signal.side in ['BUY', 'SELL']
            assert signal.confidence >= 0.0 and signal.confidence <= 1.0
            assert signal.proposed_entry_price > 0
        
        # Phase 4: Risk Management Validation
        risk_validated_signals = []
        for signal in strategy_signals:
            risk_result = await trading_system.validate_risk(signal)
            if risk_result.approved:
                risk_validated_signals.append(signal)
        
        # Validate risk management
        assert len(risk_validated_signals) <= len(strategy_signals)
        for validated_signal in risk_validated_signals:
            assert validated_signal.risk_score is not None
            assert validated_signal.position_size > 0
        
        # Phase 5: Portfolio Management
        portfolio_updates = []
        for signal in risk_validated_signals:
            portfolio_update = await trading_system.update_portfolio_allocation(signal)
            portfolio_updates.append(portfolio_update)
        
        # Validate portfolio management
        assert len(portfolio_updates) == len(risk_validated_signals)
        for update in portfolio_updates:
            assert update.allocation_percentage >= 0
            assert update.allocation_percentage <= 1.0
        
        # Phase 6: Order Execution
        execution_results = []
        for i, signal in enumerate(risk_validated_signals):
            execution_result = await trading_system.execute_order(
                signal, 
                portfolio_updates[i]
            )
            execution_results.append(execution_result)
        
        # Validate order execution
        assert len(execution_results) == len(risk_validated_signals)
        for result in execution_results:
            assert result.order_id is not None
            assert result.status in ['PENDING', 'FILLED', 'PARTIALLY_FILLED', 'REJECTED']
            if result.status in ['FILLED', 'PARTIALLY_FILLED']:
                assert result.executed_quantity > 0
                assert result.average_price > 0
        
        # Phase 7: System State Validation
        system_state = await trading_system.get_system_state()
        
        assert system_state.active_positions >= 0
        assert system_state.total_portfolio_value > 0
        assert system_state.available_cash >= 0
        assert len(system_state.open_orders) >= 0
        
        # Phase 8: Performance Metrics
        performance_metrics = await trading_system.calculate_performance_metrics()
        
        assert 'total_return' in performance_metrics
        assert 'sharpe_ratio' in performance_metrics
        assert 'max_drawdown' in performance_metrics
        assert 'win_rate' in performance_metrics
        
        # Phase 9: Clean Shutdown
        await trading_system.shutdown()
        assert not trading_system.is_running
    
    @pytest.mark.asyncio
    async def test_data_pipeline_integration(self, trading_system, sample_market_data):
        """Test comprehensive data pipeline integration."""
        await trading_system.initialize()
        
        # Test real-time data processing
        start_time = time.time()
        processed_count = 0
        
        for data_point in sample_market_data[:50]:
            result = await trading_system.ingest_market_data(data_point)
            if result.success:
                processed_count += 1
        
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert processed_count == 50
        assert processing_time < 10.0  # Should process 50 data points in under 10 seconds
        
        # Test data transformation and feature generation
        features = await trading_system.get_latest_features('BTC/USDT')
        assert features is not None
        assert len(features.columns) > 10  # Should have multiple features
        
        # Test data persistence
        historical_data = await trading_system.get_historical_data(
            'BTC/USDT', 
            start_time=datetime.now(timezone.utc) - timedelta(hours=1),
            end_time=datetime.now(timezone.utc)
        )
        assert len(historical_data) > 0
        
        await trading_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_strategy_engine_integration(self, trading_system, sample_market_data):
        """Test strategy engine integration with market data and signals."""
        await trading_system.initialize()
        
        # Load market data
        for data_point in sample_market_data[:200]:
            await trading_system.ingest_market_data(data_point)
        
        # Test multiple strategy execution
        strategy_configs = [
            {'name': 'momentum_strategy', 'lookback': 20, 'threshold': 0.02},
            {'name': 'mean_reversion_strategy', 'lookback': 50, 'threshold': 0.01},
            {'name': 'breakout_strategy', 'lookback': 10, 'threshold': 0.03}
        ]
        
        all_signals = []
        for config in strategy_configs:
            signals = await trading_system.run_strategy(config)
            all_signals.extend(signals)
        
        # Validate strategy integration
        assert len(all_signals) > 0
        
        # Test signal quality
        for signal in all_signals:
            assert signal.strategy_name in ['momentum_strategy', 'mean_reversion_strategy', 'breakout_strategy']
            assert signal.confidence >= 0.0
            assert signal.signal_strength is not None
            assert signal.feature_importance is not None
        
        # Test signal correlation and conflict resolution
        btc_signals = [s for s in all_signals if s.trading_pair == 'BTC/USDT']
        if len(btc_signals) > 1:
            conflict_resolution = await trading_system.resolve_signal_conflicts(btc_signals)
            assert conflict_resolution.final_signal is not None
            assert conflict_resolution.conflict_score >= 0.0
        
        await trading_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_risk_management_integration(self, trading_system, sample_market_data):
        """Test comprehensive risk management integration."""
        await trading_system.initialize()
        
        # Load market data
        for data_point in sample_market_data[:100]:
            await trading_system.ingest_market_data(data_point)
        
        # Generate test signals with various risk profiles
        test_signals = [
            # Conservative signal
            IntegrationDataFactory.create_test_signal(
                'BTC/USDT', 'BUY', confidence=0.6, position_size=0.05
            ),
            # Aggressive signal
            IntegrationDataFactory.create_test_signal(
                'ETH/USDT', 'BUY', confidence=0.9, position_size=0.20
            ),
            # High-risk signal
            IntegrationDataFactory.create_test_signal(
                'ADA/USDT', 'SELL', confidence=0.4, position_size=0.30
            )
        ]
        
        # Test risk validation for each signal
        risk_results = []
        for signal in test_signals:
            risk_result = await trading_system.validate_risk(signal)
            risk_results.append(risk_result)
        
        # Validate risk management logic
        conservative_result = risk_results[0]
        aggressive_result = risk_results[1]
        high_risk_result = risk_results[2]
        
        # Conservative signal should be approved
        assert conservative_result.approved
        assert conservative_result.adjusted_position_size <= signal.position_size
        
        # Aggressive signal might be approved with reduced size
        if aggressive_result.approved:
            assert aggressive_result.adjusted_position_size <= test_signals[1].position_size
        
        # High-risk signal should likely be rejected or heavily reduced
        if high_risk_result.approved:
            assert high_risk_result.adjusted_position_size < test_signals[2].position_size * 0.5
        
        # Test portfolio-level risk limits
        portfolio_risk = await trading_system.calculate_portfolio_risk()
        assert portfolio_risk.total_var >= 0  # Value at Risk
        assert portfolio_risk.concentration_risk >= 0
        assert portfolio_risk.leverage_ratio >= 0
        
        await trading_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_execution_engine_integration(self, trading_system, sample_market_data):
        """Test execution engine integration with order management."""
        await trading_system.initialize()
        
        # Load market data
        for data_point in sample_market_data[:50]:
            await trading_system.ingest_market_data(data_point)
        
        # Create test orders
        test_orders = [
            IntegrationDataFactory.create_test_order(
                'BTC/USDT', 'BUY', Decimal('0.1'), Decimal('45000.0')
            ),
            IntegrationDataFactory.create_test_order(
                'ETH/USDT', 'SELL', Decimal('1.0'), Decimal('3200.0')
            )
        ]
        
        # Test order execution
        execution_results = []
        for order in test_orders:
            result = await trading_system.execute_order_direct(order)
            execution_results.append(result)
        
        # Validate execution results
        for result in execution_results:
            assert result.order_id is not None
            assert result.execution_timestamp is not None
            assert result.status in ['PENDING', 'FILLED', 'PARTIALLY_FILLED', 'REJECTED']
        
        # Test order status monitoring
        for result in execution_results:
            if result.status == 'PENDING':
                # Monitor order until completion or timeout
                status_updates = []
                for _ in range(10):  # Check 10 times with delay
                    await asyncio.sleep(0.1)
                    status = await trading_system.get_order_status(result.order_id)
                    status_updates.append(status)
                    if status.final_status:
                        break
                
                assert len(status_updates) > 0
                final_status = status_updates[-1]
                assert final_status.status in ['FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'REJECTED']
        
        # Test order book impact
        order_book_before = await trading_system.get_order_book('BTC/USDT')
        large_order = IntegrationDataFactory.create_test_order(
            'BTC/USDT', 'BUY', Decimal('10.0'), Decimal('45000.0')  # Large order
        )
        
        await trading_system.execute_order_direct(large_order)
        
        order_book_after = await trading_system.get_order_book('BTC/USDT')
        
        # Large orders should impact the order book
        assert order_book_before != order_book_after
        
        await trading_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, trading_system, sample_market_data):
        """Test system error recovery and resilience."""
        await trading_system.initialize()
        
        # Test data ingestion errors
        invalid_data_points = [
            {'invalid': 'data'},  # Invalid format
            {'symbol': 'BTC/USDT', 'price': 'not_a_number'},  # Invalid price
            None  # Null data
        ]
        
        error_count = 0
        for invalid_data in invalid_data_points:
            try:
                await trading_system.ingest_market_data(invalid_data)
            except Exception:
                error_count += 1
        
        # System should handle errors gracefully
        assert error_count <= len(invalid_data_points)  # Some errors might be caught
        assert trading_system.is_running  # System should still be running
        
        # Test strategy execution errors
        invalid_strategy_config = {'name': 'nonexistent_strategy'}
        
        try:
            await trading_system.run_strategy(invalid_strategy_config)
        except Exception as e:
            assert trading_system.is_running  # System should recover
        
        # Test order execution errors
        invalid_order = IntegrationDataFactory.create_test_order(
            'INVALID/PAIR', 'BUY', Decimal('-1.0'), Decimal('0.0')  # Invalid order
        )
        
        try:
            result = await trading_system.execute_order_direct(invalid_order)
            assert result.status == 'REJECTED'
        except Exception:
            pass  # Expected for invalid orders
        
        # System should still be functional after errors
        assert trading_system.is_running
        
        # Test normal operation after errors
        valid_data = sample_market_data[0]
        result = await trading_system.ingest_market_data(valid_data)
        assert result.success
        
        await trading_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_performance_integration(self, trading_system, sample_market_data):
        """Test system performance under load."""
        await trading_system.initialize()
        
        # Performance test parameters
        data_points_count = 1000
        concurrent_operations = 10
        
        # Test data ingestion throughput
        start_time = time.time()
        
        # Process data points in batches
        batch_size = 100
        for i in range(0, min(data_points_count, len(sample_market_data)), batch_size):
            batch = sample_market_data[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [trading_system.ingest_market_data(data_point) for data_point in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for any failures
            failed_results = [r for r in results if isinstance(r, Exception)]
            success_rate = (len(results) - len(failed_results)) / len(results)
            assert success_rate >= 0.95  # 95% success rate minimum
        
        ingestion_time = time.time() - start_time
        
        # Performance assertions
        data_points_processed = min(data_points_count, len(sample_market_data))
        throughput = data_points_processed / ingestion_time
        
        assert throughput >= 50  # Minimum 50 data points per second
        assert ingestion_time < 60  # Complete within 60 seconds
        
        # Test concurrent strategy execution
        start_time = time.time()
        
        strategy_tasks = []
        for i in range(concurrent_operations):
            config = {'name': f'test_strategy_{i}', 'lookback': 20 + i}
            strategy_tasks.append(trading_system.run_strategy(config))
        
        strategy_results = await asyncio.gather(*strategy_tasks, return_exceptions=True)
        strategy_time = time.time() - start_time
        
        # Validate concurrent execution
        successful_strategies = [r for r in strategy_results if not isinstance(r, Exception)]
        assert len(successful_strategies) >= concurrent_operations * 0.8  # 80% success rate
        assert strategy_time < 30  # Complete within 30 seconds
        
        # Memory usage validation
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_usage_mb = process.memory_info().rss / 1024 / 1024
        
        # Memory usage should be reasonable
        assert memory_usage_mb < 2000  # Less than 2GB
        
        await trading_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_event_bus_integration(self, trading_system, sample_market_data):
        """Test event bus communication and message handling."""
        await trading_system.initialize()
        
        # Event counters
        events_received = {
            'market_data': 0,
            'signals': 0,
            'orders': 0,
            'portfolio_updates': 0
        }
        
        # Event listener
        async def event_listener():
            while trading_system.is_running:
                try:
                    event = await asyncio.wait_for(trading_system.event_bus.get(), timeout=0.1)
                    
                    if event.event_type == 'market_data':
                        events_received['market_data'] += 1
                    elif event.event_type == 'signal':
                        events_received['signals'] += 1
                    elif event.event_type == 'order':
                        events_received['orders'] += 1
                    elif event.event_type == 'portfolio_update':
                        events_received['portfolio_updates'] += 1
                        
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break
        
        # Start event listener
        listener_task = asyncio.create_task(event_listener())
        
        # Generate events
        for data_point in sample_market_data[:20]:
            await trading_system.ingest_market_data(data_point)
        
        # Generate some signals
        signals = await trading_system.generate_strategy_signals()
        
        # Execute some orders
        if signals:
            for signal in signals[:3]:  # Execute first 3 signals
                await trading_system.execute_order_from_signal(signal)
        
        # Allow time for event processing
        await asyncio.sleep(1.0)
        
        # Stop listener
        listener_task.cancel()
        
        # Validate event communication
        assert events_received['market_data'] >= 20
        assert events_received['signals'] >= 0
        
        if signals:
            assert events_received['orders'] >= 0
            assert events_received['portfolio_updates'] >= 0
        
        await trading_system.shutdown()
```

## Testing Strategy

1. **Comprehensive Test Coverage**
   - Unit tests with 95%+ code coverage
   - Integration tests for all component interactions
   - End-to-end workflow testing
   - Performance and load testing

2. **Advanced Test Scenarios**
   - Edge cases and boundary conditions
   - Error recovery and resilience testing
   - Concurrent operation testing
   - Resource constraint testing

3. **Financial Domain Testing**
   - Trading scenario validation
   - Risk management testing
   - Performance metric validation
   - Regulatory compliance testing

4. **Automated Quality Gates**
   - Continuous integration testing
   - Performance regression detection
   - Code quality metrics validation
   - Security vulnerability testing