# Feature Engine Design Document

This document outlines the design for enhancing the Feature Engine to support Scikit-learn transformers, Pandas-based operations, and Deep Learning (DL) based feature extraction, focusing on configuration, processing logic, lifecycle management, and interaction with other components. It also discusses alignment with a target `FeatureEngineInterface` for future growth, clarifies the role of feature selection, proposes documentation and testing strategies, and concludes with a summary and strategic recommendations.

## 1. Configuration Schema for Scikit-learn Transformers
(Content from previous turn - e.g., `processing_steps` as the main configuration key)
...

## 2. Outline `FeatureEngine` Logic for Applying Transformations
(Content from previous turn - generic `calculate_features` orchestrating different step types)
...

## 3. Define Lifecycle and Management of Fitted Transformers (Scikit-learn)
(Content from previous turn)
...

## 4. Analyze Interaction with Predictor-Specific Scaling
(Content from previous turn)
...

## 5. Pandas-Based Time-Series Features
(Content from previous turn - schema integrated into `processing_steps`)
...

## 6. Deep Learning-Based Feature Extractors
(Content from previous turn - schema integrated into `processing_steps`)
...

## 7. Alignment with `FeatureEngineInterface` and Future Expansion
(Content from previous turn)
...

## 8. Feature Selection and Importance in the ML Pipeline
(Content from previous turn)
...

## 9. Documentation and Testing Strategy

This section outlines a comprehensive documentation structure and a multi-layered testing strategy for the `FeatureEngine`. The goal is to ensure the engine is understandable, maintainable, extensible, and reliable in a production environment.

### 9.1. Documentation Strategy

A robust documentation strategy is essential for effective collaboration, onboarding, and long-term maintenance of the `FeatureEngine`.

#### 9.1.1. Audience Analysis

The documentation should cater to the following key audiences:

1.  **Data Scientists / Quantitative Researchers:**
    *   **Needs:** Understand available features, how to configure feature generation, how to request new features, and how to interpret feature characteristics.
    *   **Focus:** User/Configuration Guide, Feature Dictionary.
2.  **ML Engineers / System Operators:**
    *   **Needs:** Configure and deploy the `FeatureEngine`, manage its assets (models, scalers), monitor its performance, and troubleshoot issues.
    *   **Focus:** User/Configuration Guide, parts of Developer Guide related to operational aspects.
3.  **Software Developers (maintaining/extending `FeatureEngine`):**
    *   **Needs:** Understand the internal architecture, data flows, how to add new feature types, and testing procedures.
    *   **Focus:** Developer Guide, Testing Strategy.

#### 9.1.2. User/Configuration Guide

This guide is crucial for users who configure, deploy, and use the `FeatureEngine`.

*   **A. Introduction and Core Concepts:**
    *   Purpose and role of `FeatureEngine` in the broader trading system.
    *   Key architectural concepts: `processing_steps` as the central configuration paradigm, sequential execution model, management of stateful vs. stateless operations.
    *   Overview of how features are made available to downstream consumers (e.g., `PredictionService`).
    *   Explanation of data granularity (e.g., per trading pair, per tick/bar).
*   **B. `feature_engine.processing_steps` Configuration (`config.yaml`):**
    *   **General Structure:** Detailed explanation of the `processing_steps` list and common attributes for each step (`name`, `type`, `enabled`, `input_features`/`input_columns`, output naming parameters like `output_name_override`, `output_feature_prefix`).
    *   **Detailed Configuration for Each `type`:**
        *   **`pandas_rolling_statistic` / `pandas_lag` / `pandas_rate_of_change` / `pandas_time_feature` (and similar Pandas-based types):**
            *   **Parameters:** Exhaustive list of `params` (e.g., `window`, `statistic`, `quantile_value`, `lag_periods`, `time_component`, `cyclical_encoding`).
            *   **Input:** Expected format and source of `input_feature`.
            *   **Output Naming:** Default conventions and how to customize.
            *   **Examples:** Clear YAML snippets for common use cases (e.g., calculating 20-period SMA, 1-period lag of 'close', hour-of-day with sin/cos encoding).
        *   **`sklearn_transformer` / `sklearn_selector`:**
            *   **Purpose:** Applying pre-fitted Scikit-learn transformers or selectors.
            *   **Parameters:** `transformer_class`, `params` (for stateless instantiation), `fitted_transformer_path`.
            *   **Input/Output:** `input_features` specification (including wildcards like `*` if supported for "all numeric"), `output_feature_names` strategies (e.g., `suffix_transformer_generated`, `explicit`).
            *   **Examples:** `StandardScaler`, `PolynomialFeatures`, `VarianceThreshold`.
        *   **`dl_feature_extractor`:**
            *   **Purpose:** Using pre-trained deep learning models for feature extraction.
            *   **Parameters:** `model_path`, `scaler_path`, `params` (detailing `model_architecture_type`, `framework`, `input_columns`, `sequence_length`).
            *   **Input:** Sequence creation from `input_columns`.
            *   **Output Naming:** `output_feature_prefix`, `output_feature_names`.
            *   **Examples:** LSTM autoencoder, 1D CNN extractor.
    *   **C. Management of Pre-fitted Assets:**
        *   **Asset Types:** Scalers, Scikit-learn transformers/selectors, DL models.
        *   **Training & Serialization:** Best practices for offline training and saving these assets (e.g., using `joblib` for Scikit-learn, `SavedModel` for TensorFlow, `.pt` for PyTorch).
        *   **Versioning & Storage:** Recommendations for version controlling assets alongside their training code and configurations (e.g., Git LFS, DVC, dedicated artifact repository). Clear path specification in `config.yaml`.
    *   **D. Feature Naming Conventions and Output Structure:**
        *   Detailed explanation of how output feature names are derived or constructed for each step type.
        *   Description of the final data structure published by `FeatureEngine` (e.g., dictionary for real-time, DataFrame for historical).
    *   **E. Configuration Versioning:**
        *   Strategy for versioning `config.yaml` files.
        *   How changes in configuration (and thus features) are tracked and communicated, especially if downstream models have dependencies on specific feature sets.
    *   **F. Troubleshooting Guide:**
        *   Common errors (e.g., misconfiguration, missing assets, data type mismatches).
        *   Debugging tips (e.g., enabling verbose logging, testing steps in isolation).

#### 9.1.3. Developer Guide

This guide is for software developers responsible for maintaining or extending the `FeatureEngine`.

*   **A. System Architecture:**
    *   Detailed overview of the `FeatureEngine` class, its main methods (e.g., `calculate_features`, `_apply_<type>_feature`), and data flow.
    *   Explanation of internal data structures (e.g., `self.available_features`).
    *   Interaction with data providers (e.g., `OhlcvHistoryProvider`).
*   **B. Adding New Feature Calculators/Transformer Types:**
    *   **Interface/Pattern:** Define the expected interface or abstract base class for new calculator modules/functions if applicable.
    *   **Registration:** How a new `type` is registered within the `FeatureEngine`'s dispatch mechanism.
    *   **Implementation Guidelines:**
        *   **Data Access:** How to correctly access input data from `self.available_features` or underlying historical data.
        *   **Parameter Handling:** How to parse and use parameters from the `step_config['params']` dictionary.
        *   **Output Generation:** Adhering to output naming conventions and adding new features to `self.available_features`.
        *   **Error Handling:** Implementing robust error handling and logging within the new calculator.
        *   **Performance Considerations:** Writing efficient code, especially for operations on large series/frames. Avoiding unnecessary data duplication.
        *   **Stateless vs. Stateful:** Guidelines for managing state if the new calculator requires it (though pre-fitted assets are preferred for stateful operations).
*   **C. Logging, Monitoring, and Metrics:**
    *   Standard logging practices to be followed.
    *   Key metrics to expose for monitoring `FeatureEngine` health and performance (e.g., calculation latency per step, number of features generated, error rates).
*   **D. Contribution and Coding Standards:**
    *   Code style, review process, and version control (Git) practices.

#### 9.1.4. Feature Dictionary/Glossary

A centralized, easily accessible repository of all features the engine can generate.

*   **A. Purpose:** Enable users to discover and understand available features. Crucial for data scientists selecting features for models.
*   **B. Template for Entries:** Each feature should be documented with:
    *   `Feature Name`: The final unique output name.
    *   `Description`: Clear, concise explanation of what the feature represents and its potential utility.
    *   `Calculation Source Step Name`: The `name` attribute of the `processing_steps` item that generates this feature.
    *   `Calculation Source Type`: The `type` attribute (e.g., `pandas_rolling_statistic`).
    *   `Detailed Parameters`: Specific parameters used in its calculation (e.g., `window: 20`, `statistic: "mean"`, `transformer_class: "StandardScaler"`).
    *   `Input Features`: List of features directly used to compute this feature.
    *   `Data Type`: e.g., `float`, `int`, `bool`.
    *   `Expected Characteristics`: e.g., range of values (if known), presence of NaNs (e.g., at start of series for rolling features), stationarity (if applicable).
    *   `Dependencies`: Any external libraries or assets (e.g., specific version of a DL model).
*   **C. Generation and Maintenance:**
    *   **Semi-automated:** Develop scripts to parse the `feature_engine.processing_steps` from `config.yaml` to generate a base list of features and their configurations.
    *   **Manual Enrichment:** Descriptions, expected characteristics, and utility notes would likely require manual input from data scientists or developers.
    *   **Version Control:** The feature dictionary should be version-controlled alongside the codebase and configuration.
    *   **Accessibility:** Published in an easily searchable format (e.g., HTML, PDF, internal wiki).

#### 9.1.5. Documentation Management

*   **Tools:**
    *   **Source:** Markdown files stored in the Git repository alongside the code.
    *   **Generation:** Sphinx for Python-based projects, or other static site generators (MkDocs, Jekyll).
    *   **Hosting:** ReadtheDocs, GitHub Pages, Confluence, or internal documentation portal.
*   **Practices:**
    *   Documentation updates as part of the development lifecycle (e.g., new features require corresponding docs).
    *   Regular reviews of documentation for accuracy and completeness.
    *   Versioning of documentation to align with software releases.

### 9.2. Testing Strategy

A multi-layered testing strategy is critical for ensuring the `FeatureEngine` is robust, correct, and reliable.

#### 9.2.1. Testing Levels

*   **A. Unit Tests:**
    *   **Scope:** Focus on individual, isolated components or functions, primarily the logic within each `_apply_<type>_feature` method or standalone utility.
    *   **Responsibilities:**
        *   Verify correctness of calculations for a specific feature type (e.g., a rolling standard deviation, a specific Scikit-learn transformation, output of a mocked DL model).
        *   Validate output names based on configuration.
        *   Test handling of edge cases: empty inputs, inputs with NaNs, insufficient data for a calculation (e.g., window > series length).
        *   Mock external dependencies like file system access (for loading models/scalers), complex library calls, or actual DL model inference.
    *   **Examples:**
        *   For `pandas_rolling_statistic`: Given a fixed Pandas Series and window/statistic, assert the output Series is as expected. Test with `min_periods`.
        *   For `sklearn_transformer`: Provide a small DataFrame, a mock (or simple pre-fitted) transformer object, and verify the transformed DataFrame structure and values. Test different `output_feature_names` strategies.
        *   For `dl_feature_extractor`: Mock the model and scaler loading. Provide a sample input sequence and verify that the (mocked) model's output is correctly named and integrated.
*   **B. Integration Tests:**
    *   **Scope:** Test the interaction between different components of the `FeatureEngine`, particularly the execution of a sequence of `processing_steps`.
    *   **Responsibilities:**
        *   Verify that features generated in one step are correctly used as inputs in subsequent steps.
        *   Ensure the end-to-end feature calculation pipeline for a given configuration produces a consistent and correct set of features.
        *   Test data flow through the `self.available_features` DataFrame.
        *   Validate the final output structure containing all features.
    *   **Examples:**
        *   A test case with a `config.yaml` defining 3-4 `processing_steps` (e.g., raw data -> rolling mean -> scale mean -> polynomial on scaled mean). Provide initial raw data and verify the final feature values.
        *   Test configurations that use the `*` wildcard for `input_features` to ensure it correctly selects all available numeric features at that stage.
*   **C. Contract Tests (Consumer-Driven Contracts or Output Validation):**
    *   **Scope:** Ensure the output of `FeatureEngine` (e.g., the schema of the `FEATURES_CALCULATED` event or the structure of the `FeatureVector`) adheres to the expectations of downstream consumers like `PredictionService`.
    *   **Responsibilities:**
        *   Validate the names, data types, and overall structure of the features being published.
        *   If a formal schema (e.g., Avro, JSON Schema) is used for events, validate the output against this schema.
    *   **Implementation:** Can be done by having consumers publish their expected schemas, or by maintaining a shared schema definition.
*   **D. Performance Tests (Optional but Recommended):**
    *   **Scope:** Benchmark the feature generation latency and resource (CPU, memory) usage.
    *   **Responsibilities:**
        *   Identify performance bottlenecks in specific feature calculations or the overall pipeline.
        *   Establish baseline performance metrics and track regressions.
    *   **Implementation:** Use profiling tools. Run tests with realistic data volumes and configurations. Test under concurrent load if applicable.

#### 9.2.2. Test Data Management

*   **Small, Fixed Datasets:** For most unit and integration tests, use small, well-defined datasets (e.g., CSV files, hardcoded Pandas DataFrames) that cover normal inputs and specific edge cases. These should be version-controlled with the test code.
*   **Synthetic Data Generation:** Develop scripts to generate synthetic data for specific scenarios that are hard to find in real data (e.g., series with all NaNs, perfectly correlated features, data causing division by zero).
*   **Snapshot/Golden Testing:** For complex transformations or entire pipeline outputs, store "golden" versions of expected output files/data structures. Tests compare current output against these golden files. Updates to golden files must be intentional and reviewed.
*   **Test Data for Pre-fitted Assets:** For testing steps that load pre-fitted assets, include miniature, valid versions of these assets (e.g., a scaler fitted on 2 features, a tiny saved DL model that just performs a simple operation) in the test resources.

#### 9.2.3. Mocking Strategies

*   **External File System:** Use libraries like `pyfakefs` or `unittest.mock.patch` to mock file system operations when testing loading of `fitted_transformer_path`, `model_path`, etc.
*   **Complex Library Functions:** If a feature relies on a complex external library function whose behavior is hard to control, mock that function to return predictable outputs for test cases.
*   **DL Model Inference:** For `dl_feature_extractor` unit tests, mock the `model.predict()` or `model()` call to return a fixed, known feature vector. This avoids needing actual DL model files or GPU resources for most unit tests.
*   **Data Providers:** Mock any external data providers (e.g., `OhlcvHistoryProvider`) to return controlled data slices.

#### 9.2.4. CI/CD Integration

*   **Automated Execution:** All unit and integration tests should be automatically executed in the CI/CD pipeline upon every code change (commit/merge).
*   **Build Failure:** A failing test must fail the build and prevent deployment.
*   **Test Coverage:** Monitor test coverage (e.g., using `coverage.py`) and set targets to ensure adequate testing.
*   **Performance tests** might be run nightly or on-demand due to their longer execution time.

#### 9.2.5. Regression Testing

*   **Comprehensive Test Suite:** The combination of unit and integration tests serves as the primary defense against regressions.
*   **Bug-Driven Tests:** When a bug is found, a new test case that specifically reproduces the bug should be added before fixing it. This ensures the bug does not reappear.
*   **Golden Test Sets:** Regularly review and update golden test files/snapshots as features evolve legitimately.

#### 9.2.6. Data Validation Tests (Input Data Handling)

*   **Scope:** Test how the `FeatureEngine` reacts to problematic input market data.
*   **Responsibilities:**
    *   Behavior with missing OHLCV columns.
    *   Handling unexpected NaNs or infinite values in critical input fields.
    *   Resilience to sudden changes in data scale or type (if not explicitly handled by design).
    *   Logging of data quality issues.
*   **Implementation:** These might be specific integration tests where the input data is intentionally corrupted or malformed.

By implementing this comprehensive documentation and testing strategy, the `FeatureEngine` can achieve a high degree of reliability, maintainability, and trustworthiness, crucial for its role in a production trading system.
```

## 10. Summary, Strategic Recommendations, and Conclusion

This final section encapsulates the key findings from the analysis, summarizes the proposed design for an enhanced `FeatureEngine`, and provides strategic recommendations for its development and evolution.

### 10.1. Summary of Key Findings and Design Outputs

**A. Initial Analysis Findings:**
*   The original `feature_engine.py` had a foundational structure for calculating technical indicators but showed areas for improvement in configuration flexibility, extensibility to new feature types (e.g., order book, ML-based), and explicit management of parameters.
*   A significant finding was the hardcoding of parameters (e.g., RSI period) in `feature_engine.py`, bypassing the `config.yaml` values, which needs immediate correction.
*   `prediction_service.py` correctly utilized `model_feature_names` from its configuration to select features, demonstrating a clear pattern for how models consume specific feature subsets.

**B. Core Components of the Enhanced Feature Engineering System (as designed in this document):**

1.  **Unified `processing_steps` Configuration Model:**
    *   A flexible list in `config.yaml` where each item defines a feature generation or transformation step.
    *   Each step has a `name`, `type` (e.g., `pandas_rolling_statistic`, `sklearn_transformer`, `dl_feature_extractor`), `params` specific to its type, `input_features`/`input_columns`, output naming strategy, and an `enabled` flag.
2.  **Integration of Pandas-Based Features (Section 5):**
    *   Schema and logic for common time-series operations: rolling window statistics, lags/differences, and time-based cyclical features, directly configurable via `processing_steps`.
3.  **Integration of Scikit-learn Transformers & Selectors (Sections 1, 8.3):**
    *   Schema and logic for applying pre-fitted Scikit-learn objects (transformers like `StandardScaler`, `PolynomialFeatures`; selectors like `VarianceThreshold`) to features. Emphasizes loading serialized objects (e.g., via Joblib).
4.  **Conceptual Design for Deep Learning-Based Feature Extractors (Section 6):**
    *   Schema and logic for incorporating pre-trained DL models (LSTMs, CNNs) to generate learned feature embeddings from time-series sequences. Includes management of model and scaler paths.
5.  **Strategy for Alignment with `FeatureEngineInterface` (Section 7):**
    *   A phased approach to evolve the `FeatureEngine`, potentially adopting `FeatureSpec`-like structures and categorized calculation logic, to eventually align with or fully implement the `FeatureEngineInterface` for better modularity and handling of diverse data types (e.g., sentiment, macro).
6.  **Clarified Role of Feature Selection (Section 8):**
    *   Positioned feature selection primarily as an offline, model-specific process during research and training. `FeatureEngine` generates a superset of features, and `PredictionService` supplies the selected subset to models. Optional static pre-filtering in `FeatureEngine` is possible but recommended for limited use cases.
7.  **Documentation and Testing Strategy (Section 9):**
    *   Proposed structure for user/configuration guides, developer extension guides, and a feature dictionary.
    *   Outlined a unit testing strategy covering configuration, individual calculators, core engine logic, and error handling.

### 10.2. Strategic Recommendations for Further Development

**A. Immediate Priorities:**

1.  **Resolve Configuration Mismatch:** Correct `feature_engine.py` to ensure it dynamically uses parameters (e.g., RSI period, MA windows) from `config.yaml` instead of hardcoded values. This is crucial for basic operational correctness.
2.  **Implement Foundational `processing_steps` Framework:**
    *   Refactor `feature_engine.py` to use the `processing_steps` list from `config.yaml` as the primary driver for feature calculation.
    *   Implement the basic dispatch logic to different `_apply_<type>_feature` methods based on the `type` field in each step.
    *   Start with migrating existing technical indicators (SMA, RSI, MACD) to this new configurable structure (likely as `pandas_technical_indicator` or similar types).

**B. Iterative Implementation Roadmap for New Capabilities:**

1.  **Pandas-Based Features (Section 5):** Implement support for `pandas_rolling_statistic`, `pandas_lag`, `pandas_rate_of_change`, and `pandas_time_feature` types. This leverages existing Pandas capabilities and provides immediate value.
2.  **Scikit-learn Transformers & Selectors (Sections 1, 8.3):**
    *   Implement the `sklearn_transformer` and `sklearn_selector` types.
    *   Develop offline pipelines/scripts for fitting and serializing these Scikit-learn objects (e.g., `StandardScaler`, `VarianceThreshold`) on representative datasets. These are critical dependencies.
3.  **Deep Learning Feature Extractors (Section 6):**
    *   Implement the `dl_feature_extractor` type.
    *   Concurrently, significant effort will be needed for offline training, tuning, and serialization of DL models (autoencoders, CNNs) and their associated input scalers. This is a research-intensive task.
4.  **Phased Alignment with `FeatureEngineInterface` (Section 7):**
    *   Begin by enriching the `processing_steps` items with more `FeatureSpec`-like metadata (Phase 1).
    *   Consider organizing `processing_steps` by category in the config or refactoring the calculation loop (Phase 2) as more diverse features are added.
    *   A full implementation of `FeatureEngineInterface` (Phase 3) should be viewed as a longer-term goal, particularly when the need to integrate fundamentally different raw data sources (news, macro data) becomes a priority.

**C. Long-Term Architectural Vision:**

*   Continuously evaluate the benefits of moving towards a stricter implementation of `FeatureEngineInterface`, especially when requirements for handling diverse data (sentiment, macro-economic, on-chain data) and more complex feature interdependencies arise. This interface promotes modularity and scalability.

**D. Cross-Cutting Recommendations:**

1.  **Robust Unit Testing:** Implement unit tests rigorously at each stage of development, covering individual feature calculators, configuration loading, core engine logic, and error handling as outlined in Section 9.2.
2.  **Comprehensive Documentation:** Maintain and expand documentation (user guides, developer guides, feature dictionaries) as new features and capabilities are added (Section 9.1).
3.  **MLOps for Pre-fitted Assets:** Establish clear processes for training, versioning, storing, and managing all pre-fitted assets (scalers, selectors, DL models). This is critical for reproducibility and operational stability.
4.  **Performance Monitoring & Optimization:** As the number and complexity of features grow, continuously monitor the performance (latency, resource usage) of the `FeatureEngine`. Optimize critical paths as needed.
5.  **Data Quality Monitoring:** Implement checks for the quality of input data (OHLCV, etc.), as feature calculations are sensitive to this.

### 10.3. Promising Libraries for Hybrid Feature Engineering Approach

The proposed hybrid design leverages several powerful Python libraries:

*   **Pandas:** Core library for time-series data manipulation, OHLCV data handling, and implementing many custom feature calculations (rolling statistics, lags, differences, time components).
*   **Scikit-learn:** Essential for data preprocessing (scaling, encoding), feature transformation (polynomials, PCA), and static feature selection (variance threshold). Its pipeline and model persistence (via Joblib) are key.
*   **pandas-ta (or similar technical analysis libraries like TA-Lib):** Useful for efficiently calculating a wide range of standard technical indicators. Can be wrapped or used by `pandas_...` type steps.
*   **TensorFlow / PyTorch:** The primary frameworks for developing, training, and deploying the Deep Learning models (LSTMs, CNNs) used in `dl_feature_extractor` steps.
*   **Joblib (or `pickle` / `cloudpickle`):** For serializing and deserializing pre-fitted Scikit-learn objects (scalers, transformers, selectors).

### 10.4. Concluding Statement

The initial analysis revealed that while `feature_engine.py` provides a starting point, its capabilities can be significantly improved and expanded by adopting a more configurable, extensible, and hybrid approach to feature engineering.

The comprehensive design detailed in this document—integrating declarative Pandas operations, pre-fitted Scikit-learn transformers, and advanced Deep Learning-based feature extractors, all managed via a unified `processing_steps` configuration—achieves this. This hybrid model allows data scientists and ML engineers to leverage the strengths of different techniques for different feature engineering tasks, from simple moving averages to complex learned embeddings.

Furthermore, the proposed phased alignment with the `FeatureEngineInterface` provides a strategic path towards a highly modular and scalable system capable of incorporating an even wider array of data sources and feature types in the future. By implementing this design with robust testing and documentation, Gal-Friday2's `FeatureEngine` can become a powerful, flexible, and central component of its intelligent trading capabilities.
```
