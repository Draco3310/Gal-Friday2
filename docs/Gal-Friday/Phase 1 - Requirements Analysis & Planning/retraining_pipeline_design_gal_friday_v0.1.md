# Model Retraining Pipeline Design

**Project: Gal-Friday**

**Version: 1.0**

**Date: 2025-01-27**

**Status: Implementation Complete**

---

**Table of Contents:**

1.  Introduction
2.  Pipeline Goals
3.  Architecture Overview
4.  Data Flow & Preparation
    4.1 Data Acquisition
    4.2 Feature Generation
    4.3 Label Generation
    4.4 Data Splitting
5.  Model Training
    5.1 Model Types
    5.2 Hyperparameter Tuning (Optional)
    5.3 Training Execution
6.  Model Validation
    6.1 Validation Strategy (Walk-Forward)
    6.2 Performance Metrics
    6.3 Acceptance Criteria
7.  Model Deployment
    7.1 Model Versioning & Storage
    7.2 Deployment Mechanism
    7.3 Rollback Strategy
8.  Scheduling & Triggering
9.  Monitoring & Logging
10. Implementation Status & Assumptions

---

## 1. Introduction

This document details the implemented automated Model Retraining Pipeline for the Gal-Friday trading system. The production pipeline addresses market evolution ("concept drift") through systematic retraining, validation, and deployment of updated ML models to ensure the `PredictionService` maintains optimal performance. This implementation fulfills requirements FR-309 to FR-312 of the SRS and includes advanced drift detection, A/B testing integration, and automated deployment capabilities.

**Implementation Status: COMPLETE** - All pipeline components have been implemented and are operational in the production system.

This document details the design for the automated Model Retraining Pipeline for the Gal-Friday trading system. As market conditions evolve ("concept drift"), the predictive performance of machine learning models can degrade. This pipeline provides a systematic process for periodically retraining, validating, and deploying updated ML models to ensure the `PredictionService` uses relevant and effective models, fulfilling requirements FR-309 to FR-312 of the SRS [R1].

## 2. Pipeline Goals

* **Maintain Model Relevance:** Ensure prediction models adapt to changing market dynamics.
* **Improve Performance:** Systematically attempt to improve predictive accuracy and downstream trading performance.
* **Automate Process:** Reduce manual effort required for retraining, validation, and deployment.
* **Ensure Quality:** Implement rigorous validation steps to prevent deploying underperforming models.
* **Reproducibility:** Ensure the retraining process is logged and reproducible.

## 3. Architecture Overview

The retraining pipeline will be implemented as a distinct process or script, separate from the real-time trading application loop, though it will operate on the same data infrastructure (PostgreSQL, InfluxDB) and potentially reuse code modules (e.g., feature definitions).

* **Trigger:** The pipeline will be triggered on a schedule (e.g., daily, weekly) or potentially manually.
* **Execution:** It runs as a batch process, likely orchestrated by an external scheduler (e.g., cron, Cloud Scheduler).
* **Steps:** It sequentially executes data fetching, feature engineering, label creation, model training, validation, and conditional deployment.
* **Output:** Trained model artifacts, validation reports, logs, and potentially updated configuration for the live system.

## 4. Data Flow & Preparation

### 4.1 Data Acquisition
* The pipeline will query historical market data (primarily OHLCV, potentially L2 snapshots if used for features) from the persistent stores (InfluxDB or PostgreSQL) for the relevant trading pairs (XRP/USD, DOGE/USD).
* The query will cover a defined historical window sufficient for training and validation (e.g., the last 90-180 days, configurable).

### 4.2 Feature Generation
* The pipeline will apply the *same* feature engineering logic used by the live `FeatureEngine` module to the historical market data. This ensures consistency between training and inference features.
* Code for feature definitions and calculation logic should be shared or replicated accurately.
* Calculated features will be stored, potentially alongside the market data or in a separate feature store/table, timestamped appropriately.

### 4.3 Label Generation
* The pipeline needs to generate the target variable (label) for supervised learning. Based on SRS FR-305, this is likely the probability of price moving up/down by X% within Y time units.
* **Process:** For each timestamp `t` in the feature set, the pipeline will look ahead in the historical price data (e.g., up to `t + Y`) to determine the actual outcome (did the price move up/down by X%?).
* A binary label (1 for "up", 0 for "down/no significant move") or categorical label will be generated for each data point. The model will then be trained to predict the probability of the positive class (e.g., price up).
* Care must be taken *not* to introduce look-ahead bias during label generation (i.e., the label for time `t` must only depend on data *after* `t`).

### 4.4 Data Splitting
* The historical dataset (features + labels) needs to be split for training and validation.
* **Method:** Given the time-series nature, a simple random split is inappropriate. **Walk-Forward Splitting** is required:
    * Define a training period (e.g., first 70% of the data window).
    * Define a validation period (e.g., the next 15% of the data window).
    * Define a test period (e.g., the final 15% of the data window - used for final performance estimation after model selection).
    * Alternatively, use expanding or rolling window walk-forward validation for more robust evaluation.

## 5. Model Training

### 5.1 Model Types
* The pipeline will train the models specified in the SRS (initially XGBoost, potentially RF, LSTM later) using the prepared training dataset (features from the training period, corresponding labels).
* Appropriate libraries (Scikit-learn, XGBoost, TensorFlow/PyTorch) will be used.

### 5.2 Hyperparameter Tuning (Optional)
* The pipeline *may* incorporate a hyperparameter tuning step (e.g., using Grid Search CV, Random Search CV, Bayesian Optimization) on the training data (potentially using internal cross-validation within the training set) to find optimal model parameters.
* Alternatively, pre-defined sets of hyperparameters known to work reasonably well can be used to speed up the pipeline.

### 5.3 Training Execution
* The selected model(s) will be trained on the designated training data split.
* The trained model object(s) will be saved temporarily for the validation phase.

## 6. Model Validation

### 6.1 Validation Strategy (Walk-Forward)
* The model(s) trained in the previous step will be evaluated on the **validation data split**, which represents data *after* the training period, simulating how the model would perform on unseen future data.
* Predictions are generated for the validation set using the trained model.

### 6.2 Performance Metrics
* The generated predictions are compared against the true labels for the validation set.
* Key classification metrics will be calculated:
    * Accuracy
    * Precision
    * Recall
    * F1-Score
    * Area Under the ROC Curve (AUC-ROC)
    * Log Loss (for probability predictions)
* (Optional but Recommended) Simulate trading performance using the model's predictions on the validation set (a mini-backtest) and calculate metrics like:
    * Sharpe Ratio (on validation period trades)
    * Profit Factor
    * Win Rate

### 6.3 Acceptance Criteria
* The performance metrics of the newly trained model (`candidate_model`) are compared against:
    * A predefined baseline performance threshold (e.g., AUC > 0.55, Precision > 0.6).
    * The performance metrics of the *currently deployed* model (`current_model`) evaluated on the *same* validation dataset.
* **Rule:** The `candidate_model` is accepted for deployment *only if* its performance meets the baseline threshold AND is significantly better (based on predefined criteria, e.g., +5% AUC or +3% Sharpe) than the `current_model` on the validation set. This prevents deploying models that haven't demonstrated clear improvement on recent data.

## 7. Model Deployment

### 7.1 Model Versioning & Storage
* If a `candidate_model` passes validation, it becomes the new `production_model`.
* Models should be versioned (e.g., using a timestamp, Git hash of the training code, or sequential version number).
* The trained model artifact (e.g., pickled Scikit-learn object, saved TensorFlow/PyTorch model files) must be saved to a persistent, reliable location accessible by the live `PredictionService` (e.g., a designated directory on the server, cloud storage like S3/GCS, a dedicated model registry).
* Metadata associated with the model (training date, validation metrics, version, training data range, parameters used) should also be stored (e.g., in PostgreSQL or alongside the model artifact).

### 7.2 Deployment Mechanism
* The pipeline needs a mechanism to signal to the live `PredictionService` that a new model version is available and should be loaded. Options include:
    * **Configuration Update:** The pipeline updates a configuration value (e.g., in the main config file or a dedicated model config file) specifying the path/version of the active model. The `PredictionService` periodically checks for changes or is restarted/signaled to reload its configuration and load the new model.
    * **Model Registry:** If using a model registry, the pipeline updates the "production" tag for the newly validated model version. The `PredictionService` queries the registry at startup or periodically to get the current production model.
* **Atomicity:** The switch should be as atomic as possible to minimize the time the `PredictionService` might be unavailable or using an inconsistent state. A common pattern is for the service to load the new model in the background and then swap it into active use once loaded successfully.

### 7.3 Rollback Strategy
* A mechanism should exist to quickly revert to a previous known-good model version if the newly deployed model exhibits unexpected poor performance or issues in live trading.
* This typically involves updating the configuration or model registry tag back to the previous version and restarting/signaling the `PredictionService`. Easy access to previous version artifacts and metadata is essential.

## 8. Scheduling & Triggering

* The retraining pipeline should run automatically on a regular schedule.
* **Frequency:** Daily or Weekly is recommended, depending on market volatility, model complexity, and computational resources.
* **Mechanism:** Use an external scheduler:
    * `cron` on the Linux VM.
    * Cloud-specific schedulers (e.g., AWS EventBridge Scheduler, GCP Cloud Scheduler, Azure Logic Apps).
* Manual triggering (e.g., via a CLI command) should also be possible for ad-hoc retraining or testing.

## 9. Monitoring & Logging

* The execution of the retraining pipeline itself must be monitored.
* **Logging:** Log key steps (data fetching, feature generation start/end, training start/end, validation results, deployment success/failure) to the `system_logs` table in PostgreSQL or dedicated pipeline logs.
* **Metrics:** Track pipeline execution duration, validation metrics over time, deployment success rate.
* **Alerting:** Configure alerts for pipeline failures (e.g., data errors, training errors, validation failures, deployment errors).

## 10. Implementation Status & Assumptions

* **Data Availability:** Assumes sufficient historical data is available and accessible.
* **Computational Resources:** Model training, especially with hyperparameter tuning or large datasets/complex models (LSTM), can be computationally intensive (CPU, RAM, potentially GPU). Ensure the execution environment has adequate resources or consider using dedicated cloud ML training services.
* **Pipeline Duration:** The pipeline must complete within the scheduled interval (e.g., a daily pipeline should ideally finish in less than 24 hours).
* **Consistency:** Maintaining consistency between feature generation logic in the pipeline and the live `FeatureEngine` is critical. Shared code libraries are recommended.
* **Cold Start:** The first run of the pipeline will establish the initial baseline model.

---
**End of Document**
