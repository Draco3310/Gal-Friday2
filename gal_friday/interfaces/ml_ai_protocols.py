"""ML/AI protocols for model lifecycle, training, prediction, and evaluation.

This module provides protocol definitions for ML/AI operations, enabling
type-safe model management and composable ML pipeline architectures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray
    import pandas as pd

    from gal_friday.model_lifecycle.enums import ModelStage, ModelStatus
    from gal_friday.strategy_arbitrator import PredictionField


@runtime_checkable
class ModelTrainingProtocol(Protocol):
    """Protocol for model training operations.

    This protocol defines the interface for training ML models with
    support for different training strategies and configurations.
    """

    async def train(
        self,
        x_train: NDArray[np.float64] | pd.DataFrame,
        y_train: NDArray[np.float64] | pd.Series[Any],
        x_val: NDArray[np.float64] | pd.DataFrame | None = None,
        y_val: NDArray[np.float64] | pd.Series[Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train the model with provided data.

        Args:
            x_train: Training features.
            y_train: Training targets.
            x_val: Validation features (optional).
            y_val: Validation targets (optional).
            **kwargs: Additional training parameters.

        Returns:
            Dictionary containing training metrics and results.
        """
        ...

    async def fit(
        self,
        x: NDArray[np.float64] | pd.DataFrame,
        y: NDArray[np.float64] | pd.Series[Any],
        **kwargs: Any,
    ) -> Any:
        """Fit the model to training data.

        Args:
            x: Training features.
            y: Training targets.
            **kwargs: Additional fit parameters.

        Returns:
            The fitted model instance.
        """
        ...

    async def partial_fit(
        self,
        x: NDArray[np.float64] | pd.DataFrame,
        y: NDArray[np.float64] | pd.Series[Any],
        **kwargs: Any,
    ) -> Any:
        """Incrementally fit the model to training data.

        Args:
            x: Training features.
            y: Training targets.
            **kwargs: Additional fit parameters.

        Returns:
            The model instance.
        """
        ...

    def is_fitted(self) -> bool:
        """Check if the model has been fitted.

        Returns:
            True if model is fitted, False otherwise.
        """
        ...

    def get_training_parameters(self) -> dict[str, Any]:
        """Get current training parameters.

        Returns:
            Dictionary of training parameters.
        """
        ...

    async def set_training_parameters(self, params: dict[str, Any]) -> None:
        """Set training parameters.

        Args:
            params: Dictionary of training parameters to set.
        """
        ...

    def get_training_history(self) -> dict[str, Any]:
        """Get training history and metrics.

        Returns:
            Dictionary containing training history.
        """
        ...


@runtime_checkable
class ModelPredictionProtocol(Protocol):
    """Protocol for model prediction operations.

    This protocol defines the interface for making predictions with
    trained models, supporting both batch and streaming predictions.
    """

    async def predict(
        self,
        x: NDArray[np.float64] | pd.DataFrame,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """Make predictions on input data.

        Args:
            x: Input features.
            **kwargs: Additional prediction parameters.

        Returns:
            Array of predictions.
        """
        ...

    async def predict_proba(
        self,
        x: NDArray[np.float64] | pd.DataFrame,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """Make probability predictions on input data.

        Args:
            x: Input features.
            **kwargs: Additional prediction parameters.

        Returns:
            Array of probability predictions.
        """
        ...

    async def predict_batch(
        self,
        batch_data: Iterator[NDArray[np.float64] | pd.DataFrame],
        **kwargs: Any,
    ) -> AsyncIterator[NDArray[np.float64]]:
        """Make predictions on batch data.

        Args:
            batch_data: Iterator of input batches.
            **kwargs: Additional prediction parameters.

        Yields:
            Predictions for each batch.
        """
        ...

    async def predict_streaming(
        self,
        stream_data: AsyncIterator[NDArray[np.float64] | pd.DataFrame],
        **kwargs: Any,
    ) -> AsyncIterator[NDArray[np.float64]]:
        """Make predictions on streaming data.

        Args:
            stream_data: Async iterator of input data.
            **kwargs: Additional prediction parameters.

        Yields:
            Predictions for each data point.
        """
        ...

    def get_prediction_confidence(self, predictions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get confidence scores for predictions.

        Args:
            predictions: Array of predictions.

        Returns:
            Array of confidence scores.
        """
        ...

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        ...


@runtime_checkable
class ModelEvaluationProtocol(Protocol):
    """Protocol for model evaluation operations.

    This protocol defines the interface for evaluating model performance
    with various metrics and validation strategies.
    """

    async def evaluate(
        self,
        x_test: NDArray[np.float64] | pd.DataFrame,
        y_test: NDArray[np.float64] | pd.Series[Any],
        metrics: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Evaluate model performance on test data.

        Args:
            x_test: Test features.
            y_test: Test targets.
            metrics: List of metrics to compute.
            **kwargs: Additional evaluation parameters.

        Returns:
            Dictionary of evaluation metrics.
        """
        ...

    async def cross_validate(
        self,
        x: NDArray[np.float64] | pd.DataFrame,
        y: NDArray[np.float64] | pd.Series[Any],
        cv: int = 5,
        metrics: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """Perform cross-validation evaluation.

        Args:
            x: Input features.
            y: Target values.
            cv: Number of cross-validation folds.
            metrics: List of metrics to compute.
            **kwargs: Additional evaluation parameters.

        Returns:
            Dictionary of metric scores for each fold.
        """
        ...

    async def validate_model(
        self,
        validation_data: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Validate model with comprehensive validation suite.

        Args:
            validation_data: Dictionary containing validation datasets.
            **kwargs: Additional validation parameters.

        Returns:
            Dictionary containing validation results.
        """
        ...

    def get_supported_metrics(self) -> list[str]:
        """Get list of supported evaluation metrics.

        Returns:
            List of supported metric names.
        """
        ...

    async def compute_metric(
        self,
        y_true: NDArray[np.float64] | pd.Series[Any],
        y_pred: NDArray[np.float64],
        metric: str,
        **kwargs: Any,
    ) -> float:
        """Compute a specific evaluation metric.

        Args:
            y_true: True target values.
            y_pred: Predicted values.
            metric: Metric name to compute.
            **kwargs: Additional metric parameters.

        Returns:
            Computed metric value.
        """
        ...

    def get_evaluation_report(self) -> dict[str, Any]:
        """Get comprehensive evaluation report.

        Returns:
            Dictionary containing detailed evaluation report.
        """
        ...


@runtime_checkable
class ModelLifecycleProtocol(Protocol):
    """Protocol for model lifecycle management.

    This protocol defines the interface for managing the complete
    lifecycle of ML models from training to deployment.
    """

    async def create_model(
        self,
        model_config: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """Create a new model instance.

        Args:
            model_config: Model configuration parameters.
            **kwargs: Additional creation parameters.

        Returns:
            Model ID of the created model.
        """
        ...

    async def save_model(
        self,
        model_id: str,
        path: Path | str,
        **kwargs: Any,
    ) -> None:
        """Save model to storage.

        Args:
            model_id: Model identifier.
            path: Storage path for the model.
            **kwargs: Additional save parameters.
        """
        ...

    async def load_model(
        self,
        model_id: str,
        path: Path | str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Load model from storage.

        Args:
            model_id: Model identifier.
            path: Storage path for the model.
            **kwargs: Additional load parameters.

        Returns:
            Loaded model instance.
        """
        ...

    async def delete_model(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> bool:
        """Delete model from storage.

        Args:
            model_id: Model identifier.
            **kwargs: Additional delete parameters.

        Returns:
            True if deletion successful, False otherwise.
        """
        ...

    async def get_model_metadata(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get model metadata.

        Args:
            model_id: Model identifier.
            **kwargs: Additional parameters.

        Returns:
            Dictionary containing model metadata.
        """
        ...

    async def update_model_metadata(
        self,
        model_id: str,
        metadata: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Update model metadata.

        Args:
            model_id: Model identifier.
            metadata: Updated metadata dictionary.
            **kwargs: Additional parameters.
        """
        ...

    async def promote_model(
        self,
        model_id: str,
        target_stage: ModelStage,
        **kwargs: Any,
    ) -> bool:
        """Promote model to a different stage.

        Args:
            model_id: Model identifier.
            target_stage: Target stage for promotion.
            **kwargs: Additional promotion parameters.

        Returns:
            True if promotion successful, False otherwise.
        """
        ...

    async def get_model_status(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> ModelStatus:
        """Get current model status.

        Args:
            model_id: Model identifier.
            **kwargs: Additional parameters.

        Returns:
            Current model status.
        """
        ...


@runtime_checkable
class ModelRegistryProtocol(Protocol):
    """Protocol for model registry operations.

    This protocol defines the interface for managing a centralized
    registry of ML models with versioning and metadata.
    """

    async def register_model(
        self,
        model_name: str,
        model_version: str,
        model_artifact: Any,
        metadata: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """Register a new model in the registry.

        Args:
            model_name: Name of the model.
            model_version: Version of the model.
            model_artifact: Model artifact/object.
            metadata: Model metadata.
            **kwargs: Additional registration parameters.

        Returns:
            Model ID of the registered model.
        """
        ...

    async def get_model(
        self,
        model_name: str,
        model_version: str | None = None,
        stage: ModelStage | None = None,
        **kwargs: Any,
    ) -> Any:
        """Get model from registry.

        Args:
            model_name: Name of the model.
            model_version: Specific version (optional).
            stage: Model stage (optional).
            **kwargs: Additional parameters.

        Returns:
            Model artifact/object.
        """
        ...

    async def list_models(
        self,
        model_name: str | None = None,
        stage: ModelStage | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """List models in the registry.

        Args:
            model_name: Filter by model name (optional).
            stage: Filter by model stage (optional).
            **kwargs: Additional parameters.

        Returns:
            List of model metadata dictionaries.
        """
        ...

    async def get_model_versions(
        self,
        model_name: str,
        **kwargs: Any,
    ) -> list[str]:
        """Get available versions for a model.

        Args:
            model_name: Name of the model.
            **kwargs: Additional parameters.

        Returns:
            List of available model versions.
        """
        ...

    async def get_latest_model(
        self,
        model_name: str,
        stage: ModelStage | None = None,
        **kwargs: Any,
    ) -> Any:
        """Get latest model version.

        Args:
            model_name: Name of the model.
            stage: Model stage (optional).
            **kwargs: Additional parameters.

        Returns:
            Latest model artifact/object.
        """
        ...

    async def search_models(
        self,
        query: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search models in the registry.

        Args:
            query: Search query parameters.
            **kwargs: Additional parameters.

        Returns:
            List of matching model metadata dictionaries.
        """
        ...


@runtime_checkable
class ModelDeploymentProtocol(Protocol):
    """Protocol for model deployment operations.

    This protocol defines the interface for deploying ML models
    to production environments with monitoring and management.
    """

    async def deploy_model(
        self,
        model_id: str,
        deployment_config: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """Deploy model to production environment.

        Args:
            model_id: Model identifier.
            deployment_config: Deployment configuration.
            **kwargs: Additional deployment parameters.

        Returns:
            Deployment ID.
        """
        ...

    async def undeploy_model(
        self,
        deployment_id: str,
        **kwargs: Any,
    ) -> bool:
        """Undeploy model from production environment.

        Args:
            deployment_id: Deployment identifier.
            **kwargs: Additional parameters.

        Returns:
            True if undeployment successful, False otherwise.
        """
        ...

    async def get_deployment_status(
        self,
        deployment_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get deployment status.

        Args:
            deployment_id: Deployment identifier.
            **kwargs: Additional parameters.

        Returns:
            Dictionary containing deployment status.
        """
        ...

    async def update_deployment(
        self,
        deployment_id: str,
        config_updates: dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """Update deployment configuration.

        Args:
            deployment_id: Deployment identifier.
            config_updates: Configuration updates.
            **kwargs: Additional parameters.

        Returns:
            True if update successful, False otherwise.
        """
        ...

    async def get_active_deployments(
        self,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Get active deployments.

        Args:
            model_name: Filter by model name (optional).
            **kwargs: Additional parameters.

        Returns:
            List of active deployment information.
        """
        ...

    async def get_deployment_metrics(
        self,
        deployment_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get deployment performance metrics.

        Args:
            deployment_id: Deployment identifier.
            **kwargs: Additional parameters.

        Returns:
            Dictionary containing deployment metrics.
        """
        ...


@runtime_checkable
class ModelMonitoringProtocol(Protocol):
    """Protocol for model monitoring operations.

    This protocol defines the interface for monitoring deployed models
    for performance, drift, and other quality metrics.
    """

    async def start_monitoring(
        self,
        model_id: str,
        monitoring_config: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """Start monitoring for a deployed model.

        Args:
            model_id: Model identifier.
            monitoring_config: Monitoring configuration.
            **kwargs: Additional parameters.

        Returns:
            Monitoring job ID.
        """
        ...

    async def stop_monitoring(
        self,
        monitoring_job_id: str,
        **kwargs: Any,
    ) -> bool:
        """Stop monitoring for a model.

        Args:
            monitoring_job_id: Monitoring job identifier.
            **kwargs: Additional parameters.

        Returns:
            True if stopping successful, False otherwise.
        """
        ...

    async def get_monitoring_status(
        self,
        monitoring_job_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get monitoring status.

        Args:
            monitoring_job_id: Monitoring job identifier.
            **kwargs: Additional parameters.

        Returns:
            Dictionary containing monitoring status.
        """
        ...

    async def get_drift_metrics(
        self,
        model_id: str,
        time_range: tuple[datetime, datetime],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get model drift metrics.

        Args:
            model_id: Model identifier.
            time_range: Time range for metrics.
            **kwargs: Additional parameters.

        Returns:
            Dictionary containing drift metrics.
        """
        ...

    async def get_performance_metrics(
        self,
        model_id: str,
        time_range: tuple[datetime, datetime],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get model performance metrics.

        Args:
            model_id: Model identifier.
            time_range: Time range for metrics.
            **kwargs: Additional parameters.

        Returns:
            Dictionary containing performance metrics.
        """
        ...

    async def detect_anomalies(
        self,
        model_id: str,
        data: NDArray[np.float64] | pd.DataFrame,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Detect anomalies in model inputs/outputs.

        Args:
            model_id: Model identifier.
            data: Input data to analyze.
            **kwargs: Additional parameters.

        Returns:
            List of detected anomalies.
        """
        ...

    async def get_monitoring_report(
        self,
        model_id: str,
        time_range: tuple[datetime, datetime],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get comprehensive monitoring report.

        Args:
            model_id: Model identifier.
            time_range: Time range for report.
            **kwargs: Additional parameters.

        Returns:
            Dictionary containing monitoring report.
        """
        ...


@runtime_checkable
class ModelRetrainingProtocol(Protocol):
    """Protocol for model retraining operations.

    This protocol defines the interface for automated model retraining
    based on performance degradation or data drift.
    """

    async def schedule_retraining(
        self,
        model_id: str,
        retraining_config: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """Schedule model retraining.

        Args:
            model_id: Model identifier.
            retraining_config: Retraining configuration.
            **kwargs: Additional parameters.

        Returns:
            Retraining job ID.
        """
        ...

    async def trigger_retraining(
        self,
        model_id: str,
        trigger_reason: str,
        **kwargs: Any,
    ) -> str:
        """Trigger immediate model retraining.

        Args:
            model_id: Model identifier.
            trigger_reason: Reason for retraining.
            **kwargs: Additional parameters.

        Returns:
            Retraining job ID.
        """
        ...

    async def get_retraining_status(
        self,
        retraining_job_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get retraining job status.

        Args:
            retraining_job_id: Retraining job identifier.
            **kwargs: Additional parameters.

        Returns:
            Dictionary containing retraining status.
        """
        ...

    async def cancel_retraining(
        self,
        retraining_job_id: str,
        **kwargs: Any,
    ) -> bool:
        """Cancel retraining job.

        Args:
            retraining_job_id: Retraining job identifier.
            **kwargs: Additional parameters.

        Returns:
            True if cancellation successful, False otherwise.
        """
        ...

    async def get_retraining_history(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Get retraining history for a model.

        Args:
            model_id: Model identifier.
            **kwargs: Additional parameters.

        Returns:
            List of retraining job information.
        """
        ...

    async def set_retraining_triggers(
        self,
        model_id: str,
        triggers: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Set retraining triggers for a model.

        Args:
            model_id: Model identifier.
            triggers: Retraining trigger configuration.
            **kwargs: Additional parameters.
        """
        ...

    async def get_retraining_triggers(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get retraining triggers for a model.

        Args:
            model_id: Model identifier.
            **kwargs: Additional parameters.

        Returns:
            Dictionary containing retraining triggers.
        """
        ...


@runtime_checkable
class MLPipelineProtocol(Protocol):
    """Protocol for ML pipeline operations.

    This protocol defines the interface for managing end-to-end
    ML pipelines with multiple stages and components.
    """

    async def create_pipeline(
        self,
        pipeline_config: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """Create a new ML pipeline.

        Args:
            pipeline_config: Pipeline configuration.
            **kwargs: Additional parameters.

        Returns:
            Pipeline ID.
        """
        ...

    async def run_pipeline(
        self,
        pipeline_id: str,
        input_data: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run ML pipeline.

        Args:
            pipeline_id: Pipeline identifier.
            input_data: Input data for the pipeline.
            **kwargs: Additional parameters.

        Returns:
            Pipeline execution results.
        """
        ...

    async def get_pipeline_status(
        self,
        pipeline_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get pipeline status.

        Args:
            pipeline_id: Pipeline identifier.
            **kwargs: Additional parameters.

        Returns:
            Dictionary containing pipeline status.
        """
        ...

    async def stop_pipeline(
        self,
        pipeline_id: str,
        **kwargs: Any,
    ) -> bool:
        """Stop pipeline execution.

        Args:
            pipeline_id: Pipeline identifier.
            **kwargs: Additional parameters.

        Returns:
            True if stopping successful, False otherwise.
        """
        ...

    async def get_pipeline_results(
        self,
        pipeline_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get pipeline execution results.

        Args:
            pipeline_id: Pipeline identifier.
            **kwargs: Additional parameters.

        Returns:
            Dictionary containing pipeline results.
        """
        ...

    async def get_pipeline_metrics(
        self,
        pipeline_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get pipeline performance metrics.

        Args:
            pipeline_id: Pipeline identifier.
            **kwargs: Additional parameters.

        Returns:
            Dictionary containing pipeline metrics.
        """
        ...

    async def list_pipelines(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """List available ML pipelines.

        Args:
            **kwargs: Additional parameters.

        Returns:
            List of pipeline information.
        """
        ...

    async def delete_pipeline(
        self,
        pipeline_id: str,
        **kwargs: Any,
    ) -> bool:
        """Delete ML pipeline.

        Args:
            pipeline_id: Pipeline identifier.
            **kwargs: Additional parameters.

        Returns:
            True if deletion successful, False otherwise.
        """
        ...


# Import additional types for prediction interpreter protocol


@runtime_checkable
class PredictionInterpreterProtocol(Protocol):
    """Protocol for prediction interpretation and validation.

    This protocol defines the interface for components that interpret ML model
    predictions and convert them into actionable trading signals or decisions.
    """

    async def interpret(self, prediction: dict[str, Any], config: PredictionField) -> Any:
        """Interpret a prediction value according to configuration.

        Args:
            prediction: Dictionary containing prediction values
            config: Configuration for interpreting the prediction

        Returns:
            Interpreted prediction value (e.g., "BUY", "SELL", "HOLD", confidence score)
        """
        ...

    def validate(self, prediction: dict[str, Any], config: PredictionField) -> bool:
        """Validate prediction against configuration.

        Args:
            prediction: Dictionary containing prediction values
            config: Configuration for validating the prediction

        Returns:
            True if prediction is valid according to config, False otherwise
        """
        ...


@runtime_checkable
class PredictionAnalyzerProtocol(Protocol):
    """Protocol for prediction analysis and statistics.

    This protocol defines the interface for components that analyze prediction
    quality, accuracy, and provide statistical insights.
    """

    def analyze_accuracy(
        self,
        predictions: list[dict[str, Any]],
        actual_outcomes: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Analyze prediction accuracy against actual outcomes.

        Args:
            predictions: List of prediction dictionaries
            actual_outcomes: List of actual outcome dictionaries

        Returns:
            Dictionary containing accuracy metrics
        """
        ...

    def calculate_prediction_confidence(self, prediction: dict[str, Any]) -> float:
        """Calculate overall confidence score for a prediction.

        Args:
            prediction: Dictionary containing prediction values

        Returns:
            Confidence score between 0.0 and 1.0
        """
        ...

    def get_prediction_summary(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """Get summary statistics for a batch of predictions.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Dictionary containing summary statistics
        """
        ...

    def detect_prediction_anomalies(
        self,
        predictions: list[dict[str, Any]],
        threshold: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Detect anomalous predictions based on statistical analysis.

        Args:
            predictions: List of prediction dictionaries
            threshold: Anomaly detection threshold (default: 2.0 standard deviations)

        Returns:
            List of anomalous prediction dictionaries
        """
        ...


@runtime_checkable
class PredictionValidatorProtocol(Protocol):
    """Protocol for comprehensive prediction validation.

    This protocol defines the interface for components that perform thorough
    validation of predictions before they are used for trading decisions.
    """

    def validate_data_quality(self, prediction: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate prediction data quality.

        Args:
            prediction: Dictionary containing prediction values

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        ...

    def validate_value_ranges(
        self,
        prediction: dict[str, Any],
        expected_ranges: dict[str, tuple[float, float]],
    ) -> tuple[bool, list[str]]:
        """Validate prediction values are within expected ranges.

        Args:
            prediction: Dictionary containing prediction values
            expected_ranges: Dictionary mapping field names to (min, max) tuples

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        ...

    def validate_model_consistency(
        self,
        prediction: dict[str, Any],
        previous_predictions: list[dict[str, Any]],
    ) -> tuple[bool, list[str]]:
        """Validate prediction consistency with recent predictions.

        Args:
            prediction: Current prediction dictionary
            previous_predictions: List of recent prediction dictionaries

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        ...

    def get_validation_rules(self) -> dict[str, Any]:
        """Get current validation rules configuration.

        Returns:
            Dictionary containing validation rules
        """
        ...

    def update_validation_rules(self, rules: dict[str, Any]) -> None:
        """Update validation rules configuration.

        Args:
            rules: New validation rules to apply
        """
        ...
