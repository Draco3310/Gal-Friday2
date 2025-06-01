# FeatureEngine Configuration Guide

## Introduction

The `FeatureEngine` is a core component of Gal-Friday responsible for transforming raw market data (OHLCV, L2 order book snapshots, individual trades) into meaningful features. These features can then be used by machine learning models for market prediction or analysis. The engine employs a flexible system based on Scikit-learn pipelines.

This guide explains the two-tiered configuration system:
1.  **The Feature Registry (`config/feature_registry.yaml`):** A central catalog of canonical feature definitions.
2.  **Application Configuration (`config.yaml`):** Used to activate and customize features from the registry for a specific bot instance.

## 1. The Feature Registry (`config/feature_registry.yaml`)

Canonical feature definitions are stored in `config/feature_registry.yaml`. This registry acts as a centralized, versioned catalog of all available features, promoting reusability and standardized definitions.

### Purpose of the Registry
*   **Centralization:** Provides a single source of truth for how features are calculated and processed by default.
*   **Versioning:** Allows tracking changes to feature definitions over time.
*   **Clarity:** Separates detailed feature definitions from the main application configuration.
*   **Reusability:** Features defined once can be easily activated and customized across different bot configurations or experiments.

### Structure of a Registry Entry
Each top-level key in `feature_registry.yaml` is a unique identifier for a feature definition (e.g., `rsi_14_default`). The value associated with this key is a dictionary detailing the feature's properties:

*   **`description`** (string, optional):
    *   A human-readable description of the feature.
    *   Example: `"Default 14-period Relative Strength Index."`

*   **`calculator_type`** (string, mandatory):
    *   Specifies the core calculation logic. This maps to an internal `_pipeline_compute_...` function in `FeatureEngine`.
    *   Examples: `"rsi"`, `"macd"`, `"l2_spread"`, `"vwap_trades"`.

*   **`input_type`** (string, mandatory):
    *   Defines the primary type of input data the feature calculator expects.
    *   Explicitly required in registry definitions for clarity.
    *   Supported values: `"close_series"`, `"ohlcv_df"`, `"l2_book_series"`, `"trades_and_bar_starts"`.

*   **`category`** (string, optional, default: `"TECHNICAL"`):
    *   Assigns the feature to a category (e.g., `"TECHNICAL"`, `"L2_ORDER_BOOK"`, `"TRADE_DATA"`). Corresponds to the `FeatureCategory` enum.

*   **`version`** (string, optional):
    *   Version of the feature definition (e.g., `"1.0"`, `"1.1_beta"`).

*   **`parameters`** (dict, optional):
    *   A dictionary of parameters specific to the `calculator_type`.
    *   Examples:
        *   For RSI: `{"period": 14}`
        *   For MACD: `{"fast": 12, "slow": 26, "signal": 9}`
        *   For L2 features: `{"levels": 5}`

*   **`imputation`** (dict or string, optional):
    *   Configures handling of NaN values in the *output* of the feature calculation. This is a final fallback, as individual calculators aim for "Zero NaN" output where possible.
    *   **Dictionary structure:** `{"strategy": "mean" | "median" | "constant", "fill_value": <float_value>}`.
    *   **String value:** `"passthrough"` (no imputation).
    *   **If omitted or `null`:** Defaults vary (e.g., RSI fills with 50, MACD with 0).

*   **`scaling`** (dict or string, optional):
    *   Configures scaling of the feature's output. `FeatureEngine` handles this; predictors expect pre-scaled features.
    *   **Dictionary structure:** `{"method": "standard" | "minmax" | "robust", "feature_range": [min, max]}` (for minmax).
    *   **String value:** `"passthrough"` (no scaling).
    *   **If omitted or `null`:** Defaults to `StandardScaler`.

*   **`output_properties`** (dict, optional):
    *   Describes expected characteristics of the feature's output.
    *   Example: `{"value_type": "float", "range": [0, 100]}`

### Example Registry Entry
```yaml
# In config/feature_registry.yaml
rsi_14_default:
  description: "Default 14-period Relative Strength Index."
  calculator_type: "rsi"
  input_type: "close_series"
  category: "TECHNICAL"
  version: "1.0"
  parameters:
    period: 14
  imputation:
    strategy: "constant"
    fill_value: 50.0
  scaling:
    method: "minmax"
    feature_range: [0, 100]
  output_properties:
    value_type: "float"
    range: [0, 100]

l2_spread_basic:
  description: "Basic L2 order book bid-ask spread (absolute and percentage)."
  calculator_type: "l2_spread"
  input_type: "l2_book_series"
  category: "L2_ORDER_BOOK"
  version: "1.1"
  parameters: {} # No specific parameters
  imputation: {"strategy": "constant", "fill_value": 0.0}
  scaling: "passthrough"
  output_properties:
    value_type: "float" # For both abs_spread and pct_spread
```

## 2. Activating and Customizing Features in Application Configuration

The main application configuration file (e.g., `config.yaml`) uses its `feature_engine.features` section to activate features from the registry and, optionally, to override their definitions.

### Activation by List
To activate features from the registry using their default definitions, list their registry keys:

```yaml
# In main app config.yaml
feature_engine:
  features:
    - rsi_14_default       # Activates rsi_14_default from feature_registry.yaml
    - l2_spread_basic      # Activates l2_spread_basic from feature_registry.yaml
```
If a key listed here is not found in the registry, a warning will be logged, and the feature will be skipped.

### Activation and Override by Dictionary
To activate features and override parts of their registry definitions, or to define ad-hoc features not in the registry:

```yaml
# In main app config.yaml
feature_engine:
  features:
    # Activate 'rsi_14_default' from registry but override some parameters
    rsi_14_default:
      description: "Custom RSI: 20-period, robust scaled." # Override description
      parameters: # Deep-merged with registry's parameters
        period: 20
      scaling: {"method": "robust"} # Override entire scaling config

    # Activate 'macd_default' from registry, no overrides (empty dict means use registry as-is)
    macd_default: {}

    # Define an ad-hoc feature not present in the registry
    # For ad-hoc features, 'calculator_type' and 'input_type' are mandatory.
    my_custom_roc:
      description: "Rate of Change over 5 periods for close prices."
      calculator_type: "roc"
      input_type: "close_series"
      category: "TECHNICAL"
      parameters: {"period": 5}
      imputation: {"strategy": "constant", "fill_value": 0.0}
      scaling: "passthrough"
      # version and output_properties can also be defined
```

**Key Points for Overrides:**
*   The top-level key in the `features` dictionary (e.g., `rsi_14_default`, `my_custom_roc`) is the **final unique name** for the feature instance.
*   If this key matches a key in `feature_registry.yaml`, the application config acts as an override.
*   If the key does NOT match any key in the registry, it's treated as an **ad-hoc feature definition**. Ad-hoc definitions *must* provide at least `calculator_type` and `input_type`. Other fields like `parameters`, `imputation`, `scaling`, etc., should also be fully specified as needed, as there's no base definition to inherit from.

### Deep Merge Logic
When overriding a registry feature, nested dictionaries for `parameters`, `imputation`, and `scaling` are **deep-merged**.
*   For `parameters`: If the registry defines `{"period": 14, "level": 1}` and the app config provides `{"period": 20, "mode": "sma"}`, the final parameters will be `{"period": 20, "level": 1, "mode": "sma"}`.
*   For `imputation` and `scaling`: If the registry defines `{"strategy": "constant", "fill_value": 0.0}` and the app config provides `{"strategy": "mean"}`, the `fill_value` from the base is dropped, and the final config is `{"strategy": "mean"}`. If the app config provides only `{"fill_value": 0.1}` for a constant strategy, it would update just that part. Providing a string like `"passthrough"` in the override will replace the entire dictionary from the base.

Non-dictionary values (e.g., `description`, `category`, `version`, `calculator_type`, `input_type`) in the app config will directly replace the values from the registry.

## 3. Output Data Contract (Pydantic Model)

The `FeatureEngine` calculates features and publishes them via a `FEATURES_CALCULATED` event. The `features` payload within this event is structured and validated by a Pydantic model (e.g., `PublishedFeaturesV1`).

*   **Structure:** The payload is a dictionary where keys are the final "flattened" feature names (e.g., `rsi_14_default`, or `macd_default_MACD_12_26_9` for multi-column outputs like MACD lines).
*   **Data Type:** All feature values in this dictionary are **floats**. The previous string formatting has been removed.
*   **Validation:** `FeatureEngine` attempts to instantiate the Pydantic model with all calculated features. If a feature defined in the Pydantic model is missing from the calculations, or if a value is not a float (e.g. NaN for a non-optional field), Pydantic will raise a validation error, and the `FeatureEngine` will log this and skip publishing the event for that cycle. This ensures that downstream consumers like `PredictionService` receive a well-defined, type-safe feature set.

This Pydantic model serves as the data contract for features. If new features are added or existing ones produce different outputs (e.g., more columns from a DataFrame-producing feature), the Pydantic model (`PublishedFeaturesV1` or a successor) in `gal_friday/core/feature_models.py` must be updated accordingly.

---

This new approach provides a more robust, maintainable, and clear way to manage feature definitions while still allowing for flexible customization at the application level.
```
