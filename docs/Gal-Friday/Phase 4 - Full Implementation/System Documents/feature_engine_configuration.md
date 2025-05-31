# FeatureEngine Configuration Guide

## Introduction

The `FeatureEngine` is a core component of Gal-Friday responsible for transforming raw market data (OHLCV, L2 order book snapshots, individual trades) into meaningful features that can be used by machine learning models for market prediction or analysis. It employs a flexible system based on Scikit-learn pipelines, where each feature's calculation and post-processing steps are defined through a configuration.

This guide explains how to configure these features via the main application configuration file (typically a YAML file that is loaded into a Python dictionary and passed to the `FeatureEngine` during its initialization).

## General Configuration Structure

The feature definitions for the `FeatureEngine` reside under the `features` key, which itself is usually nested under a `feature_engine` key in the main application configuration.

```yaml
# Example: part of a larger config.yaml
application_name: "GalFridayTradingBot"
# ... other global configurations ...

feature_engine:
  trade_history_maxlen: 2000 # Max number of recent trades to store per trading pair
  # ... other engine-level configurations might be added here ...

  features:
    # Feature definitions for various indicators and derived data points go here.
    # Each key below is a unique name you assign to your feature configuration.
    rsi_14_default:
      calculator_type: "rsi" # Specifies using the RSI calculation logic
      parameters: {"period": 14}
      # Imputation and scaling will use defaults (e.g., fill RSI NaNs with 50, apply StandardScaler)

    macd_custom:
      calculator_type: "macd"
      parameters: {"fast": 8, "slow": 21, "signal": 5}
      imputation: {"strategy": "constant", "fill_value": 0.0}
      scaling: "passthrough" # No scaling will be applied to MACD outputs

    # ... more feature definitions ...
```

## Defining Individual Features

Each key under the `features:` block is a unique identifier for a specific feature configuration. For example, `rsi_14_default` or `my_custom_macd`. The value associated with this key is a dictionary specifying how that feature should be calculated and processed.

The following fields are recognized for each feature configuration entry:

*   **`calculator_type`** (string, mandatory or inferred):
    *   Specifies the core calculation logic to be used. This typically maps to an internal `_pipeline_compute_...` function within the `FeatureEngine`.
    *   While it can sometimes be inferred from the feature key (e.g., if "rsi" is in `rsi_14_default`), explicitly setting it is recommended for clarity.
    *   Supported values include:
        *   `"rsi"` (Relative Strength Index)
        *   `"macd"` (Moving Average Convergence Divergence)
        *   `"bbands"` (Bollinger Bands)
        *   `"roc"` (Rate of Change)
        *   `"atr"` (Average True Range)
        *   `"stdev"` (Standard Deviation of close prices)
        *   `"vwap_ohlcv"` (Volume Weighted Average Price from OHLCV data)
        *   `"l2_spread"` (Bid-Ask Spread from L2 order book)
        *   `"l2_imbalance"` (Order Book Imbalance from L2 data)
        *   `"l2_wap"` (Weighted Average Price from L2 data)
        *   `"l2_depth"` (Cumulative Depth from L2 data)
        *   `"vwap_trades"` (Volume Weighted Average Price from raw trade data)
        *   `"volume_delta"` (Difference between buy and sell volume from raw trade data)

*   **`category`** (string, optional, default: `"TECHNICAL"`):
    *   Assigns the feature to a category. This is for organizational and potential interface purposes.
    *   Common values (corresponding to `FeatureCategory` enum): `"TECHNICAL"`, `"L2_ORDER_BOOK"`, `"TRADE_DATA"`, `"SENTIMENT"`, `"CUSTOM"`.

*   **`description`** (string, optional):
    *   A user-friendly description of what this feature configuration represents.

*   **`parameters`** (dict, optional):
    *   A dictionary containing parameters specific to the chosen `calculator_type`.
    *   Examples:
        *   For RSI, ROC, StDev: `{"period": 14}`
        *   For MACD: `{"fast": 12, "slow": 26, "signal": 9}`
        *   For Bollinger Bands: `{"length": 20, "std_dev": 2.0}`
        *   For ATR, VWAP_OHLCV: `{"length": 14}`
        *   For L2 features (imbalance, wap, depth): `{"levels": 5}` (number of L2 levels to consider)
        *   For trade-based features (vwap_trades, volume_delta): `{"bar_interval_seconds": 60}` (to align with OHLCV bar duration)
    *   If parameters are not provided, sensible defaults are often used internally (e.g., period 14 for RSI). Check `FeatureEngine._build_feature_pipelines` for specific defaults.

*   **`imputation`** (dict or string, optional):
    *   Configures how NaN (Not a Number) values in the *output* of the feature calculation are handled.
    *   **Dictionary structure:** `{"strategy": "mean" | "median" | "constant", "fill_value": <float_value>}`.
        *   `"mean"`: Fill NaNs with the mean of the non-NaN values in the series/column.
        *   `"median"`: Fill NaNs with the median.
        *   `"constant"`: Fill NaNs with the value specified in `fill_value`.
    *   **Special string value:** `"passthrough"`: No output imputation is performed; NaNs will remain.
    *   **If omitted or `None`:** A default imputation strategy is applied, which varies by feature type (e.g., for RSI, it might fill with 50.0; for MACD, with 0.0; for others, with the mean of the calculated series/column).

*   **`scaling`** (dict or string, optional):
    *   Configures how the feature's output values are scaled.
    *   **Dictionary structure:** `{"method": "standard" | "minmax" | "robust", "feature_range": [min, max]}`.
        *   `"standard"`: Uses `StandardScaler` (zero mean, unit variance).
        *   `"minmax"`: Uses `MinMaxScaler`. `feature_range` (e.g., `[0, 1]` or `[-1, 1]`) is optional and defaults to `(0, 1)`.
        *   `"robust"`: Uses `RobustScaler` (handles outliers better). Optional `quantile_range` can be provided.
    *   **Special string value:** `"passthrough"`: No scaling is performed.
    *   **If omitted or `None`:** A default scaling method is applied (typically `StandardScaler`).

*   **`input_type`** (string, optional, advanced):
    *   Specifies the type of input data the feature calculator expects.
    *   This is usually inferred automatically from the `calculator_type`. For example, "rsi" implies "close_series"; "atr" implies "ohlcv_df".
    *   Manual specification is generally not needed unless for custom or advanced scenarios.
    *   Common inferred values: `"close_series"`, `"ohlcv_df"`, `"l2_book_series"`, `"trades_and_bar_starts"`.

## Examples Section

Here are a few examples of complete feature definitions:

```yaml
# In your main application config, under feature_engine:
features:
  rsi_14_custom_scaling:
    calculator_type: "rsi" # Could be inferred if "rsi" is in the key
    category: "TECHNICAL"
    parameters: {"period": 14}
    imputation: {"strategy": "constant", "fill_value": 50.0} # Impute NaNs in RSI output with 50
    scaling: {"method": "minmax", "feature_range": [0, 100]} # Scale RSI to 0-100 range
    description: "14-period RSI, imputed with 50, scaled to [0,100]."

  macd_8_21_5:
    calculator_type: "macd"
    parameters: {"fast": 8, "slow": 21, "signal": 5}
    # Imputation: Will use default for MACD (e.g., fillna(0.0) for output columns).
    # Scaling: Will use default (e.g., StandardScaler for output columns).
    description: "MACD with 8/21/5 periods, default processing."

  l2_depth_top_3:
    calculator_type: "l2_depth" # input_type will be inferred as 'l2_book_series'
    category: "L2_ORDER_BOOK"
    parameters: {"levels": 3} # Calculate depth using top 3 L2 levels
    imputation: {"strategy": "mean"} # Fill NaNs in output depth columns with mean of respective column
    scaling: "passthrough" # No scaling for depth figures
    description: "Cumulative L2 depth for top 3 levels."

  volume_delta_per_minute:
    calculator_type: "volume_delta" # input_type inferred as 'trades_and_bar_starts'
    category: "TRADE_DATA"
    parameters: {"bar_interval_seconds": 60} # Aligns with 60-second OHLCV bars
    imputation: {"strategy": "constant", "fill_value": 0.0} # If no trades, delta is 0
    scaling: {"method": "robust"} # Use RobustScaler
    description: "Volume delta over 60-second intervals."

  atr_7_passthrough:
    calculator_type: "atr" # input_type inferred as "ohlcv_df"
    parameters: {"length": 7}
    imputation: "passthrough" # NaNs from ATR calculation will remain
    scaling: "passthrough"    # ATR value will be raw
    description: "7-period ATR with no post-processing."
```

## Notes on Defaults

*   If `imputation` or `scaling` sections are omitted entirely from a feature's configuration, the `FeatureEngine` applies sensible defaults. These defaults are generally:
    *   **Imputation:** For RSI, fill with 50. For MACD and Volume Delta, fill with 0. For most other features (especially those producing DataFrames or values that can be averaged), fill with the mean of the calculated series/column. `passthrough` is also an option.
    *   **Scaling:** `StandardScaler()` is typically the default if no scaling configuration is provided. `passthrough` skips scaling.
*   It's recommended to check the `FeatureEngine._build_feature_pipelines` method's helper functions (`get_output_imputer_step`, `get_output_scaler_step`) and the specific pipeline construction logic for the most up-to-date default behaviors if relying heavily on them.

## Relation to `InternalFeatureSpec`

Internally, the `FeatureEngine` parses these YAML/dictionary configurations into `InternalFeatureSpec` objects. This dataclass provides a structured way for the engine to manage and use the feature definitions when constructing the Scikit-learn processing pipelines. Understanding the fields of `InternalFeatureSpec` (as listed in this guide) helps in creating valid and effective configurations.
```
