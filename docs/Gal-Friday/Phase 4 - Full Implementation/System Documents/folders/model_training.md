# Model Training Folder (`gal_friday/model_training`) Documentation

## Folder Overview

The `gal_friday/model_training` folder is designated within the Gal-Friday trading system's codebase to house all scripts, modules, configurations, and utilities related to the training, evaluation, and packaging of machine learning models. These models are intended for use by the `PredictionService` to generate trading predictions.

**Crucially, it must be noted that as of the current documentation date (reflecting the state of the provided source code information), this `gal_friday/model_training` folder is empty.** The subsequent sections describe its intended purpose and the types of components it would typically contain in a fully developed MLOps environment for the system.

## Intended Purpose and Contents

The `model_training` folder is envisioned to be the central hub for all activities that precede the registration of a model into the `model_lifecycle.Registry`. Its primary purpose is to ensure a structured, reproducible, and maintainable approach to developing and updating the machine learning models that drive Gal-Friday's trading decisions.

Typically, this folder would house components such as:

-   **Data Preprocessing Scripts/Modules:**
    -   Scripts or Python modules dedicated to cleaning raw market data (or other data sources like sentiment data).
    -   Transforming data into a suitable format for model consumption (e.g., creating lagged features, handling missing values beyond simple gap filling, normalizing or standardizing features).
    -   Implementing feature selection algorithms.
    -   Performing feature scaling (e.g., MinMax scaling, Standard scaling) and saving the scaler objects.
    -   Encoding categorical features if any.
    -   Splitting data into well-defined training, validation, and test sets, ensuring temporal integrity for time-series data.

-   **Model Definition Modules:**
    -   Python files containing the architectural definitions of various machine learning models. This could include:
        -   Configurations and helper functions for instantiating Scikit-learn pipelines.
        -   Code defining custom neural network architectures (e.g., LSTMs, Transformers using TensorFlow/Keras or PyTorch).
        -   Parameter setups for gradient boosting models like XGBoost or LightGBM.

-   **Training Pipelines/Scripts:**
    -   Orchestration scripts (e.g., Python scripts, Jupyter notebooks converted to scripts, or workflow tool configurations like Kubeflow Pipelines or Apache Airflow DAGs if the system scales to that complexity).
    -   These pipelines would manage the end-to-end model training process:
        -   Loading preprocessed data.
        -   Instantiating the defined model architectures.
        -   Executing training loops, including backpropagation for neural networks or fitting procedures for classical models.
        -   Implementing hyperparameter tuning strategies (e.g., grid search, random search, Bayesian optimization using libraries like Optuna or Ray Tune).
        -   Performing model evaluation on validation sets during and after training, using appropriate metrics (e.g., accuracy, precision, recall, F1-score, log loss, Sharpe ratio if evaluating a strategy proxy).
        -   Saving (serializing) the trained model artifacts (e.g., pickled Scikit-learn models, HDF5 files for Keras models, ONNX files for interoperability) and any associated objects like scalers or encoders.
        -   Generating comprehensive metadata about the training run (dataset versions, hyperparameters used, evaluation metrics, training duration, environment details).

-   **Evaluation Scripts:**
    -   Scripts dedicated to performing a more detailed and rigorous evaluation of trained models on unseen test sets.
    -   These would generate a wider array of performance metrics, visualizations (e.g., ROC curves, confusion matrices, feature importance plots), and potentially compare performance against baseline models or previous versions.

-   **Configuration Files for Training:**
    -   YAML, JSON, or Python configuration files specific to different model training runs or experiments.
    -   These might define:
        -   Paths to training/validation/test datasets.
        -   Lists of features to be used for a particular model.
        -   Hyperparameter grids or ranges for tuning.
        -   Experiment tracking parameters (e.g., for MLflow or Weights & Biases).
        -   Resource allocation for training jobs.

## Relationship with `model_lifecycle`

The `model_training` folder and the `gal_friday/model_lifecycle` folder are designed to work in close conjunction, forming a continuous MLOps loop:

-   **Output to Input:** The primary outputs of the components within the `model_training` folder are the trained `ModelArtifacts` (the serialized model files and related assets) and their corresponding `ModelMetadata` (performance metrics, training parameters, data lineage, etc.).
-   **Registration:** These outputs (artifacts and metadata) serve as the direct inputs to the `Registry` class within the `model_lifecycle.registry` module. A successfully trained and evaluated model from a `model_training` pipeline would be registered in the `Registry` to make it available for staging, experimentation, and deployment.
-   **Retraining Orchestration:** The `RetrainingPipeline` (defined in `model_lifecycle.retraining_pipeline`) would likely invoke or integrate with the training scripts or pipelines located in the `model_training` folder. When the `RetrainingPipeline` detects model drift or is triggered by a schedule, it would use the components here to execute a new training run with fresh data.

## Current Status and Future Development

As stated, the `gal_friday/model_training` folder is **currently empty**. This implies that the processes for training, evaluating, and packaging the machine learning models used by the `PredictionService` are, at present, handled either:
-   **Externally:** Models might be trained in separate environments (e.g., Jupyter notebooks, dedicated ML platforms) and their artifacts manually prepared for registration.
-   **Manually:** The process might involve manual steps by data scientists or ML engineers.

**Future Development:**
To enhance the MLOps capabilities and automation of the Gal-Friday system, future development will necessarily involve populating this folder with the components described above. This will:
-   Standardize the model development process.
-   Enable reproducible training runs.
-   Facilitate easier integration with the `model_lifecycle` management tools, particularly for automated retraining and continuous model improvement.
-   Provide a clear and organized location for all model training-related code and configurations.

Without these components, the system relies on external or ad-hoc processes for model generation and updates, which can be less efficient and harder to govern in a production trading environment.

## Adherence to Standards

A well-structured `model_training` folder, when populated, would significantly contribute to achieving reproducible, maintainable, and auditable machine learning model development. This aligns with MLOps best practices, which advocate for treating model training with the same rigor as software development, including version control for code and data, automated pipelines, and robust testing and evaluation frameworks.
