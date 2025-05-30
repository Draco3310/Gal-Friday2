# Model Lifecycle Folder (`gal_friday/model_lifecycle`) Documentation

## Folder Overview

The `gal_friday/model_lifecycle` folder provides a comprehensive MLOps (Machine Learning Operations) framework specifically designed for managing the entire lifecycle of machine learning models used within the Gal-Friday trading system. This encompasses a range of critical functions including robust model registration and versioning, secure storage of model artifacts (both locally and in the cloud), systematic experimentation and A/B testing of model variants, continuous monitoring for model drift and performance degradation, and automated retraining pipelines. The goal of this folder is to ensure that ML models deployed in Gal-Friday are high-performing, reliable, reproducible, and can be updated or rolled back efficiently.

## Key Modules and Their Roles

The `model_lifecycle` folder is composed of several key modules that work together to provide a full MLOps solution:

### `registry.py` (`Registry` class, `ModelMetadata`, `ModelArtifact`, `ModelStage`, `ModelStatus`)

-   **Purpose:** This module implements a centralized `Registry` for managing all machine learning models within the Gal-Friday system. It acts as the single source of truth for model versions, their metadata, and their current deployment stage.
-   **Key Components & Functionality:**
    -   **`ModelMetadata` (Dataclass):** Stores comprehensive information about each model version, including its unique ID, version string, model type (e.g., "xgboost", "lstm", "sklearn"), training date, training data summary (e.g., features used, date range), performance metrics on validation sets (e.g., accuracy, F1-score, AUC, Sharpe ratio if applicable), model lineage (e.g., based on which previous model or experiment), and any relevant hyperparameters.
    -   **`ModelArtifact` (Dataclass/Interface):** Represents the actual serialized model file(s) and any associated files like scalers, tokenizers, or feature encoders. It includes information about the artifact's storage location (local path or cloud URI) and checksum for integrity.
    -   **`ModelStage` (Enum):** Defines the different stages a model version can be in, such as `DEVELOPMENT`, `STAGING`, `PRODUCTION`, `ARCHIVED`. This facilitates a controlled promotion process.
    -   **`ModelStatus` (Enum):** Indicates the current operational status of a model version, e.g., `PENDING_REGISTRATION`, `REGISTERED`, `VALIDATION_FAILED`, `DEPLOYMENT_ACTIVE`.
    -   **`Registry` Class:**
        -   **Registration:** Allows new model versions (metadata and artifacts) to be registered.
        -   **Versioning:** Automatically assigns or manages version numbers for models.
        -   **Storage Management:** Coordinates with storage backends (local or cloud via `CloudStorageBackend`) to store and retrieve `ModelArtifacts`.
        -   **Metadata Persistence:** Interacts with a `ModelRepository` (from the DAL) to persist `ModelMetadata` in a relational database.
        -   **Stage Management:** Provides methods to transition models between different stages (e.g., `promote_to_production(model_id, version)`).
        -   **Model Retrieval:** Allows services (like `PredictionService`) to fetch specific model versions, often the latest model in the `PRODUCTION` stage for a given model name or purpose.
-   **Importance:** Ensures that all models are centrally tracked, versioned, and their lineage is maintained, which is critical for reproducibility, auditing, and governance.

### `experiment_manager.py` (`ExperimentManager` class, `ExperimentConfig`, `VariantPerformance`, `ExperimentStatus`, `AllocationStrategy`)

-   **Purpose:** Manages A/B testing and other forms of experimentation between different model versions or even entirely different model types targeting the same prediction task.
-   **Key Components & Functionality:**
    -   **`ExperimentConfig` (Dataclass):** Defines the setup for an experiment, including its name, description, the control model (e.g., current production model), one or more treatment/variant models, the allocation strategy for routing prediction requests, and success criteria.
    -   **`AllocationStrategy` (Enum/Interface):** Defines how prediction requests are allocated to different model variants in an experiment (e.g., `PERCENTAGE_SPLIT` like 90/10, `USER_SEGMENT`, `SHADOW_MODE` where variants run but don't influence actions).
    -   **`VariantPerformance` (Dataclass):** Stores performance metrics collected for each variant during an experiment (e.g., prediction accuracy, impact on simulated P&L if tied to a strategy).
    -   **`ExperimentStatus` (Enum):** Tracks the status of an experiment (e.g., `CREATED`, `RUNNING`, `COMPLETED_PROMOTED`, `COMPLETED_ROLLED_BACK`, `CANCELLED`).
    -   **`ExperimentManager` Class:**
        -   **Experiment Lifecycle:** Manages the creation, configuration, starting, stopping, and completion of experiments.
        -   **Request Routing:** When an experiment is active, it can intercept prediction requests (or work with `PredictionService`) to route them to the appropriate model variant based on the `AllocationStrategy`.
        -   **Performance Tracking:** Collects and aggregates performance data for each variant.
        -   **Statistical Significance:** May include logic to calculate statistical significance of performance differences between variants.
        -   **Automatic Promotion/Rollback:** Based on results and pre-defined criteria, can automatically promote a winning variant to production (by updating the `Registry`) or roll back.
        -   **Persistence:** Interacts with an `ExperimentRepository` (from the DAL) to store experiment configurations and results.
-   **Importance:** Provides a data-driven way to evaluate new models or model changes in a controlled manner before full production deployment, reducing the risk associated with model updates.

### `retraining_pipeline.py` (`RetrainingPipeline` class, `DriftDetector`, `RetrainingJob`, `DriftMetrics`, `RetrainingTrigger`, `DriftType`)

-   **Purpose:** Automates the process of monitoring production models for performance degradation or drift and orchestrates their retraining when necessary.
-   **Key Components & Functionality:**
    -   **`DriftType` (Enum):** Defines types of drift that can be monitored, such as `CONCEPT_DRIFT` (statistical properties of the target variable change), `DATA_DRIFT` (statistical properties of input features change), `PREDICTION_DRIFT` (model's output distribution changes), or `PERFORMANCE_DEGRADATION` (actual metrics like accuracy drop).
    -   **`DriftMetrics` (Dataclass):** Stores metrics used to quantify drift (e.g., p-values from statistical tests, changes in feature distributions, error rates).
    -   **`DriftDetector` Class/Interface:**
        -   Continuously (or periodically) monitors input data, predictions, and ground truth (actual outcomes) for production models.
        -   Calculates `DriftMetrics` and compares them against predefined thresholds.
        -   Signals the `RetrainingPipeline` if significant drift or degradation is detected.
    -   **`RetrainingTrigger` (Enum):** Defines how a retraining job can be initiated (e.g., `MANUAL`, `SCHEDULED`, `DRIFT_DETECTED`).
    -   **`RetrainingJob` (Dataclass):** Represents a single retraining task, including its status, trigger reason, model ID, version to be retrained, new data window, and outcome.
    -   **`RetrainingPipeline` Class:**
        -   **Orchestration:** Manages the end-to-end retraining workflow:
            -   Receives retraining triggers (manual, scheduled, or from `DriftDetector`).
            -   Fetches new training data (potentially using `HistoricalDataService`).
            -   Initiates the model training process (this might involve calling external training scripts/services or having integrated training capabilities).
            -   Validates the newly retrained model against a holdout dataset.
            -   If validation is successful, registers the new model version with the `Registry`.
            -   Optionally, can automatically promote the new model to staging or even production if criteria are met, or initiate an A/B test via `ExperimentManager`.
        -   **Persistence:** Interacts with a `RetrainingRepository` (from the DAL) to store metadata about retraining jobs and their outcomes.
-   **Importance:** Ensures that models remain accurate and relevant over time as market conditions change, by automating the detection of issues and the process of updating models.

### `cloud_storage.py` (`CloudStorageBackend` ABC, `GCSBackend`, `S3Backend`)

-   **Purpose:** Provides an abstraction layer and concrete implementations for storing and retrieving large model artifacts (serialized models, scalers, etc.) in cloud-based object storage services.
-   **Key Components & Functionality:**
    -   **`CloudStorageBackend` (Abstract Base Class):** Defines a standard interface for cloud storage operations, including methods like `upload_artifact(local_path, remote_uri)`, `download_artifact(remote_uri, local_path)`, `delete_artifact(remote_uri)`, `artifact_exists(remote_uri)`.
    -   **`GCSBackend(CloudStorageBackend)`:** Concrete implementation for Google Cloud Storage (GCS). Uses the `google-cloud-storage` client library.
    -   **`S3Backend(CloudStorageBackend)`:** Concrete implementation for Amazon S3. Uses the `boto3` client library.
    -   **Checksum Verification:** Implementations should ideally include checksum calculation (e.g., MD5, SHA256) and verification during uploads and downloads to ensure data integrity.
-   **Importance:** Facilitates scalable and reliable storage for model artifacts, which can be large. Using an abstraction allows the system to be easily configured for different cloud providers or even a local filesystem backend for development.

### `__init__.py`

-   **Purpose:** Marks the `model_lifecycle` directory as a Python package.
-   **Key Aspects:**
    -   Allows modules and their classes/functions within `model_lifecycle` to be imported using package notation (e.g., `from gal_friday.model_lifecycle.registry import Registry`).
    -   Typically exports key public classes, dataclasses, and Enums from its modules (e.g., `Registry`, `ModelMetadata`, `ModelStage`, `ExperimentManager`, `RetrainingPipeline`, `CloudStorageBackend`) to make them directly accessible at the `gal_friday.model_lifecycle` package level.

## MLOps Workflow

These components work in concert to create a cohesive MLOps pipeline for Gal-Friday:

1.  **Training & Packaging:** ML models are trained either through external processes or potentially orchestrated by the `RetrainingPipeline`. Once trained, the model files (e.g., a pickled scikit-learn model, an H5 file for a Keras model) and any associated artifacts (like feature scalers) are packaged together (conceptually as a `ModelArtifact`).
2.  **Registration:** The new model version, along with its comprehensive `ModelMetadata` (training details, performance metrics, lineage), is registered with the `Registry`. The `Registry` coordinates with a `CloudStorageBackend` (if configured) to store the physical `ModelArtifact` and persists the `ModelMetadata` in the database via `ModelRepository`.
3.  **Staging & Validation:** The newly registered model typically starts in a `DEVELOPMENT` or `STAGING` `ModelStage`. In staging, it can undergo further validation, shadow deployment, or limited live testing.
4.  **Experimentation (A/B Testing):** The `ExperimentManager` can be used to conduct A/B tests or canary releases, comparing the performance of a new model version (e.g., from `STAGING`) against the current `PRODUCTION` model. This provides data-driven evidence for promotion decisions.
5.  **Promotion to Production:** Based on validation and experimentation results, a model version can be promoted to the `PRODUCTION` stage via the `Registry`. The `PredictionService` would then query the `Registry` to load and use this production-designated model.
6.  **Monitoring & Drift Detection:** Once in production, the `DriftDetector` component of the `RetrainingPipeline` continuously monitors the model's input features, predictions, and (if available) actual outcomes against ground truth. It looks for signs of data drift, concept drift, or performance degradation.
7.  **Automated Retraining:** If significant drift or degradation is detected (or based on a schedule or manual trigger), the `RetrainingPipeline` initiates a new retraining job. This job typically involves fetching fresh data, retraining the model (using the same or an updated training process), validating the new version, and then re-inserting it into the lifecycle at the registration stage (Step 2).
8.  **Archival:** Older or poorly performing models can be moved to an `ARCHIVED` stage in the `Registry`, keeping their metadata and artifacts for historical reference but removing them from active consideration for deployment.

This structured lifecycle is paramount in a dynamic environment like financial markets. It ensures that models are not just deployed once but are continuously monitored, evaluated, and updated, maintaining their quality, reproducibility, and reliability over time.

## Interactions with other System Parts

-   **`PredictionService`**: Directly interacts with the `Registry` to load model artifacts and metadata for the models it needs to serve. If experiments are active, it might also interact with or be guided by the `ExperimentManager` to route requests to different model variants.
-   **DAL (Data Access Layer)**: The `model_lifecycle` components heavily rely on the DAL for persistence.
    -   `ModelRepository`: Stores and retrieves `ModelMetadata`.
    -   `ExperimentRepository`: Stores and retrieves `ExperimentConfig` and `VariantPerformance` data.
    -   `RetrainingRepository`: Stores and retrieves `RetrainingJob` information and `DriftMetrics`.
-   **`ConfigManager`**: Provides essential configuration settings for the `model_lifecycle` components, such as:
    -   Storage paths for local model artifacts.
    -   Credentials and bucket/container names for cloud storage backends.
    -   Default parameters for experiments and retraining pipelines.
    -   Drift detection thresholds.
-   **`HistoricalDataService`**: Used by the `RetrainingPipeline` to fetch fresh datasets for model retraining.
-   **`LoggerService`**: All `model_lifecycle` components use the `LoggerService` for detailed logging of their operations.

## Adherence to Standards

The `model_lifecycle` folder embodies MLOps best practices for managing AI/ML systems in production. By providing tools for versioning, experimentation, monitoring, and automated retraining, it aims to bring the same level of rigor and automation to machine learning model management as DevOps brings to software development. This is crucial for building trustworthy and sustainable AI-driven trading systems.
