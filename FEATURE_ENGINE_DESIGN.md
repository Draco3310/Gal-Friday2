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
(Content from previous turn)
...

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
