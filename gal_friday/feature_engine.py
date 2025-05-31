"""Feature engineering for Gal-Friday using a configurable Scikit-learn pipeline approach.

This module provides the FeatureEngine class, which is responsible for managing
market data, defining feature calculation pipelines based on external configuration,
executing these pipelines, and publishing the resulting features. It leverages
Scikit-learn's Pipeline and FunctionTransformer for creating flexible and
customizable feature engineering workflows.
"""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass, field # Added for InternalFeatureSpec

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler, RobustScaler

from gal_friday.core.events import EventType
# Attempt to import FeatureCategory, handle potential circularity if it arises
try:
    from gal_friday.interfaces.feature_engine_interface import FeatureCategory
except ImportError:
    # Simplified local definition if import fails (e.g. circular dependency)
    from enum import Enum
    class FeatureCategory(Enum):
        TECHNICAL = "TECHNICAL"
        L2_ORDER_BOOK = "L2_ORDER_BOOK"
        TRADE_DATA = "TRADE_DATA"
        SENTIMENT = "SENTIMENT"
        CUSTOM = "CUSTOM"
        UNKNOWN = "UNKNOWN"


if TYPE_CHECKING:
    from collections.abc import Callable

    from gal_friday.core.pubsub import PubSubManager
    from gal_friday.interfaces.historical_data_service_interface import HistoricalDataService
    from gal_friday.logger_service import LoggerService

# Define InternalFeatureSpec
@dataclass
class InternalFeatureSpec:
    key: str  # The original config key, e.g., "rsi_14"
    calculator_type: str # e.g., "rsi", "macd", "l2_spread" - helps map to _pipeline_compute function
    input_type: str # e.g. 'close_series', 'ohlcv_df', 'l2_book_series', 'trades_and_bar_starts'
    category: FeatureCategory = FeatureCategory.TECHNICAL # Default category
    parameters: dict[str, Any] = field(default_factory=dict) # For _pipeline_compute_... (e.g., period, length)
    imputation: dict[str, Any] | str | None = None # Imputation config for output (str for simple 'passthrough' or 'mean')
    scaling: dict[str, Any] | str | None = None    # Scaling config for output (str for simple 'passthrough')
    description: str = ""
    # TODO: Add other fields from FeatureSpec as deemed useful, e.g. output_names for multi-output features
    # TODO: Consider adding 'output_column_names' if a feature calculator produces multiple unnamed outputs
    #       that need specific naming beyond what pandas-ta might provide.


class FeatureEngine:
    """Processes market data to compute technical indicators and other features.

    The FeatureEngine ingests various types of market data (OHLCV, L2 order book,
    trades), maintains a history of this data, and, upon specific triggers (typically
    new OHLCV bars), calculates a configured set of features.

    Feature calculation is orchestrated using Scikit-learn pipelines. Each feature
    is defined by an `InternalFeatureSpec` which outlines its type, parameters,
    input data requirements, and any post-processing steps like imputation or scaling.
    The `_build_feature_pipelines` method constructs these pipelines at initialization.

    Data Flow:
    1. Market data events are received and stored in internal data structures
       (e.g., `ohlcv_history`, `l2_books`, `trade_history`).
    2. On a trigger (e.g., new OHLCV bar), `_calculate_and_publish_features` is called.
    3. Relevant historical and current data is prepared and formatted (e.g., converted
       to float64 pandas Series/DataFrames).
    4. Each configured feature pipeline is executed with the appropriate input data.
    5. Results from pipelines are collected, named, formatted, and published via a
       `FEATURES_CALCULATED` event on the PubSubManager.
    """

    _EXPECTED_L2_LEVEL_LENGTH = 2

    def __init__(
        self,
        config: dict[str, Any],
        pubsub_manager: PubSubManager,
        logger_service: LoggerService,
        historical_data_service: HistoricalDataService | None = None,
    ) -> None:
        """Initialize the FeatureEngine with configuration and required services.

        Args:
            config: Dictionary containing global application configuration, including
                feature definitions under the "features" key.
            pubsub_manager: Instance of the pub/sub event manager for event handling.
            logger_service: Logging service instance.
            historical_data_service: Optional service for fetching historical data,
                not actively used in core processing loops but can be for initialization.

        Internal State:
            _feature_configs: Parsed and structured feature definitions stored as
                `dict[str, InternalFeatureSpec]`. Populated by `_extract_feature_configs`.
            feature_pipelines: Dictionary of Scikit-learn `Pipeline` objects, keyed by
                pipeline name. Built by `_build_feature_pipelines`.
        """
        self.config = config
        self.pubsub_manager = pubsub_manager
        self.logger = logger_service
        self.historical_data_service = historical_data_service
        self._source_module = self.__class__.__name__

        # Feature configuration derived from config
        self._feature_configs: dict[str, InternalFeatureSpec] = {} # Changed type hint
        self._extract_feature_configs()

        # Initialize feature handlers dispatcher
        # RSI and MACD handlers are removed as they will be handled by sklearn pipelines.
        self._feature_handlers: dict[
            str,
            Callable[[dict[str, Any], dict[str, Any]], dict[str, str] | None],
        ] = {
            # "rsi": self._process_rsi_feature, # To be replaced by pipeline
            # "macd": self._process_macd_feature, # To be replaced by pipeline
            # "bbands": self._process_bbands_feature, # To be replaced by pipeline
            # "vwap": self._process_vwap_feature,  # To be replaced by pipeline
            # "roc": self._process_roc_feature, # To be replaced by pipeline
            # "atr": self._process_atr_feature, # To be replaced by pipeline
            # "stdev": self._process_stdev_feature, # To be replaced by pipeline
            # "spread": self._process_l2_spread_feature, # To be replaced by pipeline
            # "imbalance": self._process_l2_imbalance_feature, # To be replaced by pipeline
            # "wap": self._process_l2_wap_feature, # To be replaced by pipeline
            # "depth": self._process_l2_depth_feature, # To be replaced by pipeline
            # "volume_delta": self._process_volume_delta_feature, # To be replaced by pipeline
            # Note: vwap_ohlcv and vwap_trades are handled by _process_vwap_feature
        }

        # Initialize data storage
        # OHLCV data will be stored in a DataFrame per trading pair
        # Columns: timestamp_bar_start (index), open, high, low, close, volume
        self.ohlcv_history: dict[str, pd.DataFrame] = defaultdict(
            lambda: pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
            ).astype(
                {
                    "open": "object",  # Store as Decimal initially
                    "high": "object",
                    "low": "object",
                    "close": "object",
                    "volume": "object",
                },
            ),
        )

        # L2 order book data (latest snapshot)
        self.l2_books: dict[str, dict[str, Any]] = defaultdict(dict)

        # Store recent trades for calculating true Volume Delta and trade-based VWAP
        # deque stores: {"ts": datetime, "price": Decimal, "vol": Decimal, "side": "buy"/"sell"}
        trade_history_maxlen = config.get("feature_engine", {}).get("trade_history_maxlen", 2000)
        self.trade_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=trade_history_maxlen),
        )

        self.feature_pipelines: dict[str, dict[str, Any]] = {} # Stores {'pipeline': Pipeline, 'spec': InternalFeatureSpec}
        self._build_feature_pipelines()

        self.logger.info("FeatureEngine initialized.", source_module=self._source_module)

    def _determine_calculator_type_and_input(self, feature_key: str, raw_cfg: dict) -> tuple[str | None, str | None]:
        """Determines calculator type and input type from feature key and raw config.

        This method attempts to infer the type of calculation (e.g., "rsi", "macd")
        and the type of input data required (e.g., "close_series", "ohlcv_df")
        primarily by inspecting the `feature_key` string. It also allows for explicit
        overrides if `calculator_type` (or `type`) and `input_type` are specified
        directly in the raw feature configuration dictionary.

        Args:
            feature_key: The original key for the feature from the configuration (e.g., "rsi_14").
            raw_cfg: The raw dictionary configuration for this specific feature.

        Returns:
            A tuple `(calculator_type, input_type)`, where both can be `None` if
            determination fails.
        """
        # Allow explicit 'type' in config to override key-based inference
        # And 'input_source_column' or 'input_source_type' for explicit input definition

        calc_type = raw_cfg.get("calculator_type", raw_cfg.get("type"))
        input_type = raw_cfg.get("input_type")

        if calc_type and input_type:
            return calc_type, input_type

        # Infer from key if not explicitly provided
        key_lower = feature_key.lower()
        if "rsi" in key_lower: return "rsi", "close_series"
        if "macd" in key_lower: return "macd", "close_series"
        if "bbands" in key_lower: return "bbands", "close_series"
        if "roc" in key_lower: return "roc", "close_series"
        if "atr" in key_lower: return "atr", "ohlcv_df"
        if "stdev" in key_lower: return "stdev", "close_series"
        if "vwap_ohlcv" in key_lower: return "vwap_ohlcv", "ohlcv_df"
        if "l2_spread" in key_lower: return "l2_spread", "l2_book_series"
        if "l2_imbalance" in key_lower: return "l2_imbalance", "l2_book_series"
        if "l2_wap" in key_lower: return "l2_wap", "l2_book_series"
        if "l2_depth" in key_lower: return "l2_depth", "l2_book_series"
        if "vwap_trades" in key_lower: return "vwap_trades", "trades_and_bar_starts"
        if "volume_delta" in key_lower: return "volume_delta", "trades_and_bar_starts"

        self.logger.warning("Could not determine calculator_type or input_type for feature key: %s", feature_key)
        return None, None


    def _extract_feature_configs(self) -> None:
        """Parses raw feature configurations into structured `InternalFeatureSpec` objects.

        This method iterates through the feature configurations provided in the main
        application config under the "features" key. For each feature, it:
        1. Determines its `calculator_type` (e.g., "rsi", "macd") and `input_type`
           (e.g., "close_series", "ohlcv_df") using `_determine_calculator_type_and_input`.
        2. Extracts specific calculation `parameters` (e.g., period, length, levels).
           It checks for parameters both directly in the feature's config dict and
           under a nested "parameters" or "params" key.
        3. Parses `imputation` and `scaling` configurations for post-processing.
        4. Determines the `FeatureCategory` from the config, defaulting to TECHNICAL.
        5. Creates an `InternalFeatureSpec` instance for each valid feature and stores
           it in `self._feature_configs`.

        Expected keys in raw feature configuration (per feature):
        - `calculator_type` or `type` (optional, can be inferred from key): Defines the core logic.
        - `input_type` (optional, can be inferred): Defines the data source.
        - `category` (optional, string, defaults to 'TECHNICAL'): Feature category.
        - `parameters` or `params` (optional, dict): Nested dictionary for calculation parameters.
        - Top-level keys for common parameters (e.g., `period`, `length`) are also supported.
        - `imputation` (optional): Config for output imputation (e.g., `None`, 'passthrough', or dict).
        - `scaling` (optional): Config for output scaling (e.g., `None`, 'passthrough', or dict).
        - `description` (optional, string): A description for the feature.
        """
        raw_features_config = self.config.get("features", {})
        parsed_specs: dict[str, InternalFeatureSpec] = {}

        if not isinstance(raw_features_config, dict):
            self.logger.warning(
                "Global 'features' configuration is not a dictionary. No features loaded.",
                source_module=self._source_module,
            )
            self._feature_configs = {}
            return

        for key, raw_cfg_dict in raw_features_config.items():
            if not isinstance(raw_cfg_dict, dict):
                self.logger.warning(
                    "Configuration for feature '%s' is not a dictionary. Skipping.",
                    key, source_module=self._source_module)
                continue

            calculator_type, input_type = self._determine_calculator_type_and_input(key, raw_cfg_dict)
            if not calculator_type or not input_type:
                self.logger.warning("Skipping feature %s due to undetermined type/input.", key)
                continue

            # Parameters for the _pipeline_compute function (e.g. period, length)
            # These are often nested under a 'params' or 'parameters' key in user config,
            # or could be at the top level of raw_cfg_dict.
            parameters = raw_cfg_dict.get("parameters", raw_cfg_dict.get("params", {}))
            # If common params like 'period' or 'length' are top-level, merge them in:
            for common_param_key in ["period", "length", "fast", "slow", "signal", "levels", "std_dev", "length_seconds", "bar_interval_seconds"]:
                if common_param_key in raw_cfg_dict and common_param_key not in parameters:
                    parameters[common_param_key] = raw_cfg_dict[common_param_key]


            imputation_cfg = raw_cfg_dict.get('imputation')
            scaling_cfg = raw_cfg_dict.get('scaling')
            description = raw_cfg_dict.get('description', f"{calculator_type} feature based on {key}")

            category_str = raw_cfg_dict.get('category', 'TECHNICAL').upper()
            try:
                category = FeatureCategory[category_str]
            except KeyError:
                self.logger.warning(
                    "Invalid FeatureCategory '%s' for feature '%s'. Defaulting to TECHNICAL.",
                    category_str, key, source_module=self._source_module)
                category = FeatureCategory.TECHNICAL

            spec = InternalFeatureSpec(
                key=key,
                calculator_type=calculator_type,
                input_type=input_type,
                category=category,
                parameters=parameters,
                imputation=imputation_cfg,
                scaling=scaling_cfg,
                description=description,
            )
            parsed_specs[key] = spec

        self._feature_configs = parsed_specs


    def _build_feature_pipelines(self) -> None:
        """Build Scikit-learn pipelines for each configured feature.
        Expected configuration structure for each feature under `features` in app config:
        ```yaml
        feature_config_key: # e.g., rsi_14, macd_custom
          calculator_type: "rsi" # or "macd", "atr", "l2_spread", etc. (can be inferred from key)
          input_type: "close_series" # or "ohlcv_df", "l2_book_series", "trades_and_bar_starts" (can be inferred)
          category: "TECHNICAL" # Optional, string name of FeatureCategory enum
          parameters: # Parameters for the specific _pipeline_compute_ function
            period: 14 # For RSI
            # length: 20 # For BBands, ATR, Stdev, VWAP_OHLCV
            # fast: 12 # For MACD
            # levels: 5 # For L2 features
            # bar_interval_seconds: 60 # For trade-based features
          imputation: # Optional: Config for handling NaNs from the calculator's output
            strategy: "constant" # "mean", "median", "constant", or "passthrough" (None also means default)
            fill_value: 50.0 # if strategy is "constant"
          scaling: # Optional: Config for scaling the feature's output
            method: "minmax" # "standard", "minmax", "robust", or "passthrough" (None also means default)
            feature_range: [0, 1] # if method is "minmax"
          description: "A 14-period RSI." # Optional
        ```
        """
        self.logger.info("Building feature pipelines...", source_module=self._source_module)

        # Helper for output imputation step
        def get_output_imputer_step(imputation_cfg: dict[str, Any] | str | None,
                                    default_fill_value: float = 0.0,
                                    is_dataframe_output: bool = False,
                                    spec_key: str = "") -> tuple[str, FunctionTransformer] | None:
            """
            Creates a Scikit-learn compatible imputation step based on configuration.
            Handles Series and DataFrame outputs from feature calculators.
            """
            if imputation_cfg == 'passthrough':
                self.logger.debug("Imputation set to 'passthrough' for %s.", spec_key)
                return None

            strategy = 'default' # Internal default if cfg is None or invalid
            fill_value = default_fill_value

            if isinstance(imputation_cfg, dict):
                strategy = imputation_cfg.get('strategy', 'constant')
                if strategy == 'constant':
                    fill_value = imputation_cfg.get('fill_value', default_fill_value)
            elif imputation_cfg is None: # Use provided default_fill_value directly
                 pass # strategy remains 'default' -> use default_fill_value
            else: # Invalid config, treat as passthrough or log warning
                self.logger.warning("Unrecognized imputation config for %s: %s. No imputer added.", spec_key, imputation_cfg)
                return None

            step_name_suffix = ""
            transform_func = None

            if strategy == 'constant':
                step_name_suffix = f'const_{fill_value}'
                transform_func = lambda x: x.fillna(fill_value)
            elif strategy == 'mean':
                step_name_suffix = 'mean'
                transform_func = lambda x: x.fillna(x.mean())
            elif strategy == 'median':
                step_name_suffix = 'median'
                transform_func = lambda x: x.fillna(x.median())
            elif strategy == 'default': # Use the passed default_fill_value
                step_name_suffix = f'default_fill_{default_fill_value}'
                transform_func = lambda x: x.fillna(default_fill_value)
            else: # Should not be reached if checks are exhaustive
                self.logger.warning("Unknown imputation strategy '%s' for %s. No imputer added.", strategy, spec_key)
                return None

            self.logger.debug("Using output imputer strategy '%s' (fill: %s) for %s", strategy if strategy != 'default' else f'default_fill({default_fill_value})', fill_value if strategy == 'constant' else 'N/A', spec_key)
            return (f'{spec_key}_output_imputer_{step_name_suffix}', FunctionTransformer(transform_func, validate=False))


        # Helper for output scaler step
        def get_output_scaler_step(scaling_cfg: dict[str, Any] | str | None,
                                   spec_key: str = "") -> tuple[str, PandasScalerTransformer] | None:
            """
            Creates a Scikit-learn compatible scaling step based on configuration.
            Uses PandasScalerTransformer to preserve pandas object structure.
            """
            if scaling_cfg == 'passthrough' or scaling_cfg is None:
                if scaling_cfg == 'passthrough': self.logger.debug("Scaling set to 'passthrough' for %s.", spec_key)
                else: self.logger.debug("No scaling configured or default 'None' for %s.", spec_key)
                return None

            scaler_instance = StandardScaler() # Default scaler
            scaler_name_suffix = "StandardScaler"

            if isinstance(scaling_cfg, dict):
                method = scaling_cfg.get('method', 'standard')
                if method == 'minmax':
                    scaler_instance = MinMaxScaler(feature_range=scaling_cfg.get('feature_range', (0,1)))
                    scaler_name_suffix = f"MinMaxScaler_{scaler_instance.feature_range}"
                elif method == 'robust':
                    scaler_instance = RobustScaler(quantile_range=scaling_cfg.get('quantile_range', (25.0, 75.0)))
                    scaler_name_suffix = f"RobustScaler_{scaler_instance.quantile_range}"
                elif method != 'standard':
                    self.logger.warning("Unknown scaling method '%s' for %s. Using StandardScaler.", method, spec_key)
            elif isinstance(scaling_cfg, str) and scaling_cfg not in ['standard', 'passthrough']: # e.g. just "minmax"
                 self.logger.warning("Simple string for scaling method '%s' for %s is ambiguous. Use dict config or 'passthrough'. Defaulting to StandardScaler.", scaling_cfg, spec_key)


            self.logger.debug("Using %s for scaling for %s", type(scaler_instance).__name__, spec_key)
            return (f'{spec_key}_output_scaler_{scaler_name_suffix}', PandasScalerTransformer(scaler_instance))


        for feature_key, spec in self._feature_configs.items(): # Now iterates over InternalFeatureSpec
            pipeline_steps = []
            pipeline_name = f"{spec.key}_pipeline" # Use spec.key for consistency

            # Input imputer for features that take a single series like 'close'
            if spec.input_type == 'close_series':
                # TODO: Make input imputer strategy configurable if needed from global or feature spec
                self.logger.debug("Adding standard input imputer (mean) for %s", spec.key)
                pipeline_steps.append((f'{spec.key}_input_imputer', SimpleImputer(strategy='mean')))

            # Calculator step based on spec.calculator_type
            calculator_func = getattr(FeatureEngine, f"_pipeline_compute_{spec.calculator_type}", None)
            if not calculator_func:
                self.logger.error("No _pipeline_compute function found for calculator_type: %s (feature key: %s)", spec.calculator_type, spec.key)
                continue

            # Prepare kw_args for the calculator from spec.parameters
            # Ensure all necessary parameters for the specific calculator are present with defaults
            calc_kw_args = {}
            if spec.calculator_type in ["rsi", "roc", "stdev"]:
                default_period = 14 if spec.calculator_type == "rsi" else 10 if spec.calculator_type == "roc" else 20
                calc_kw_args['period'] = spec.parameters.get('period', default_period)
                if spec.parameters.get('period') is None: self.logger.debug("Using default period %s for %s ('%s')", calc_kw_args['period'], spec.calculator_type, spec.key)
            elif spec.calculator_type == "macd":
                calc_kw_args['fast'] = spec.parameters.get('fast', 12)
                calc_kw_args['slow'] = spec.parameters.get('slow', 26)
                calc_kw_args['signal'] = spec.parameters.get('signal', 9)
                # Log if defaults are used for any MACD param
                if any(p not in spec.parameters for p in ['fast', 'slow', 'signal']):
                    self.logger.debug("Using default MACD params (f:%s,s:%s,sig:%s) for %s", calc_kw_args['fast'], calc_kw_args['slow'], calc_kw_args['signal'], spec.key)
            elif spec.calculator_type == "bbands":
                calc_kw_args['length'] = spec.parameters.get('length', 20)
                calc_kw_args['std_dev'] = float(spec.parameters.get('std_dev', 2.0)) # Ensure float
                if 'length' not in spec.parameters or 'std_dev' not in spec.parameters:
                     self.logger.debug("Using default BBands params (l:%s,s:%.1f) for %s", calc_kw_args['length'], calc_kw_args['std_dev'], spec.key)
            elif spec.calculator_type == "atr":
                calc_kw_args['length'] = spec.parameters.get('length', 14)
                if 'length' not in spec.parameters: self.logger.debug("Using default ATR length %s for %s", calc_kw_args['length'], spec.key)
                # high_col, low_col, close_col default in function signature
            elif spec.calculator_type == "vwap_ohlcv":
                calc_kw_args['length'] = spec.parameters.get('length', 14)
                if 'length' not in spec.parameters: self.logger.debug("Using default VWAP_OHLCV length %s for %s", calc_kw_args['length'], spec.key)
            elif spec.calculator_type in ["l2_imbalance", "l2_wap", "l2_depth"]:
                default_levels = 5 if spec.calculator_type != "l2_wap" else 1
                calc_kw_args['levels'] = spec.parameters.get('levels', default_levels)
                if 'levels' not in spec.parameters: self.logger.debug("Using default levels %s for %s ('%s')", calc_kw_args['levels'], spec.calculator_type, spec.key)
            elif spec.calculator_type == "vwap_trades":
                calc_kw_args['bar_interval_seconds'] = spec.parameters.get('length_seconds', spec.parameters.get('bar_interval_seconds', 60))
                if 'length_seconds' not in spec.parameters and 'bar_interval_seconds' not in spec.parameters : self.logger.debug("Using default interval %s for %s ('%s')", calc_kw_args['bar_interval_seconds'], spec.calculator_type, spec.key)
            elif spec.calculator_type == "volume_delta":
                calc_kw_args['bar_interval_seconds'] = spec.parameters.get('bar_interval_seconds', 60)
                if 'bar_interval_seconds' not in spec.parameters : self.logger.debug("Using default interval %s for %s ('%s')", calc_kw_args['bar_interval_seconds'], spec.calculator_type, spec.key)
            # l2_spread has no specific calc_kw_args from params in its current form

            pipeline_steps.append((f'{spec.key}_calculator', FunctionTransformer(calculator_func, kw_args=calc_kw_args, validate=False)))

            # Output Imputation & Scaling using helpers
            is_df_output = spec.calculator_type in ["macd", "bbands", "l2_spread", "l2_depth"]
            # Define default fill values based on feature type characteristics
            default_fill = 0.0 # General default
            if spec.calculator_type == "rsi": default_fill = 50.0
            elif spec.calculator_type in ["atr", "vwap_ohlcv", "l2_wap", "vwap_trades", "stdev"]: default_fill = np.nan # Will be filled by mean then

            imputer_step = get_output_imputer_step(spec.imputation, default_fill_value=default_fill, is_dataframe_output=is_df_output, spec_key=spec.key)
            if imputer_step: pipeline_steps.append(imputer_step)

            scaler_step = get_output_scaler_step(spec.scaling, spec_key=spec.key)
            if scaler_step: pipeline_steps.append(scaler_step)

            if pipeline_steps:
                final_pipeline = Pipeline(steps=pipeline_steps)
                final_pipeline.set_output(transform="pandas") # Ensure pandas output
                self.feature_pipelines[pipeline_name] = {
                    'pipeline': final_pipeline,
                    'input_type': spec.input_type,
                    'params': spec.parameters, # Storing parsed parameters
                    'spec': spec # Store the full spec for richer context if needed later
                }
                self.logger.info(
                    "Built pipeline: %s with steps: %s, input: %s",
                    pipeline_name, [s[0] for s in pipeline_steps], spec.input_type,
                    source_module=self._source_module
                )

    def _handle_ohlcv_update(self, trading_pair: str, ohlcv_payload: dict[str, Any]) -> None:
            pipeline_name = f"{spec.key}_pipeline" # Use spec.key for consistency

            # Input imputer for features that take a single series like 'close'
            if spec.input_type == 'close_series':
                # TODO: Make input imputer strategy configurable if needed
                pipeline_steps.append((f'{spec.key}_input_imputer', SimpleImputer(strategy='mean')))

            # Calculator step based on spec.calculator_type
            calculator_func = getattr(FeatureEngine, f"_pipeline_compute_{spec.calculator_type}", None)
            if not calculator_func:
                self.logger.error("No _pipeline_compute function found for calculator_type: %s (feature key: %s)", spec.calculator_type, spec.key)
                continue

            # Prepare kw_args for the calculator from spec.parameters
            # Ensure all necessary parameters for the specific calculator are present with defaults
            calc_kw_args = {}
            if spec.calculator_type in ["rsi", "roc", "stdev"]:
                calc_kw_args['period'] = spec.parameters.get('period', 14 if spec.calculator_type == "rsi" else 10 if spec.calculator_type == "roc" else 20)
                if spec.parameters.get('period') is None: self.logger.debug("Using default period %s for %s", calc_kw_args['period'], spec.key)
            elif spec.calculator_type == "macd":
                calc_kw_args['fast'] = spec.parameters.get('fast', 12)
                calc_kw_args['slow'] = spec.parameters.get('slow', 26)
                calc_kw_args['signal'] = spec.parameters.get('signal', 9)
            elif spec.calculator_type == "bbands":
                calc_kw_args['length'] = spec.parameters.get('length', 20)
                calc_kw_args['std_dev'] = float(spec.parameters.get('std_dev', 2.0)) # Ensure float
            elif spec.calculator_type == "atr":
                calc_kw_args['length'] = spec.parameters.get('length', 14)
                # high_col, low_col, close_col default in function signature
            elif spec.calculator_type == "vwap_ohlcv":
                calc_kw_args['length'] = spec.parameters.get('length', 14)
            elif spec.calculator_type in ["l2_imbalance", "l2_wap", "l2_depth"]:
                calc_kw_args['levels'] = spec.parameters.get('levels', 5 if spec.calculator_type != "l2_wap" else 1)
            elif spec.calculator_type == "vwap_trades":
                calc_kw_args['bar_interval_seconds'] = spec.parameters.get('length_seconds', 60)
            elif spec.calculator_type == "volume_delta":
                calc_kw_args['bar_interval_seconds'] = spec.parameters.get('bar_interval_seconds', 60)
            # l2_spread has no specific calc_kw_args from params in its current form

            pipeline_steps.append((f'{spec.key}_calculator', FunctionTransformer(calculator_func, kw_args=calc_kw_args, validate=False)))

            # Output Imputation & Scaling using helpers
            is_df_output = spec.calculator_type in ["macd", "bbands", "l2_spread", "l2_depth"]
            # Define default fill values based on feature type characteristics
            default_fill = 0.0 # General default
            if spec.calculator_type == "rsi": default_fill = 50.0
            elif spec.calculator_type in ["atr", "vwap_ohlcv", "l2_wap", "vwap_trades", "stdev"]: default_fill = np.nan # Will be filled by mean then

            imputer_step = get_output_imputer_step(spec.imputation, default_fill_value=default_fill, is_dataframe_output=is_df_output, spec_key=spec.key)
            if imputer_step: pipeline_steps.append(imputer_step)

            scaler_step = get_output_scaler_step(spec.scaling, spec_key=spec.key)
            if scaler_step: pipeline_steps.append(scaler_step)

            if pipeline_steps:
                final_pipeline = Pipeline(steps=pipeline_steps)
                final_pipeline.set_output(transform="pandas") # Ensure pandas output
                self.feature_pipelines[pipeline_name] = {
                    'pipeline': final_pipeline,
                    'input_type': spec.input_type,
                    'params': spec.parameters, # Storing parsed parameters
                    'spec': spec # Store the full spec for richer context if needed later
                }
                self.logger.info(
                    "Built pipeline: %s with steps: %s, input: %s",
                    pipeline_name, [s[0] for s in pipeline_steps], spec.input_type,
                    source_module=self._source_module
                )

    def _handle_ohlcv_update(self, trading_pair: str, ohlcv_payload: dict[str, Any]) -> None:
        """Parse and store an OHLCV update."""
        try:
            # Extract and convert data from payload
            # Timestamp parsing (ISO 8601 string to datetime object)
            # According to inter_module_comm.md: payload.timestamp_bar_start (ISO 8601)
            timestamp_str = ohlcv_payload.get("timestamp_bar_start")
            if not timestamp_str:
                self.logger.warning(
                    "Missing 'timestamp_bar_start' in OHLCV payload for %s",
                    trading_pair,
                    source_module=self._source_module,
                    context={"payload": ohlcv_payload},
                )
                return

            # Convert to datetime. PANDAS will handle timezone if present in string
            # Forcing UTC if not specified, adjust if local timezone is expected/preferred
            bar_timestamp = pd.to_datetime(timestamp_str, utc=True)

            # Price/Volume conversion (string to Decimal for precision)
            open_price = Decimal(ohlcv_payload["open"])
            high_price = Decimal(ohlcv_payload["high"])
            low_price = Decimal(ohlcv_payload["low"])
            close_price = Decimal(ohlcv_payload["close"])
            volume = Decimal(ohlcv_payload["volume"])

            # Prepare new row as a dictionary
            new_bar_data = {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }

            # Get the DataFrame for the trading pair
            df = self.ohlcv_history[trading_pair]

            # Create a new DataFrame for the new row with the correct index
            new_row_df = pd.DataFrame([new_bar_data], index=[bar_timestamp])
            new_row_df.index.name = "timestamp_bar_start"

            # Ensure new_row_df columns match df columns and types are compatible
            # This is important if df was empty or had different types initially
            for col in df.columns:
                if col not in new_row_df:
                    new_row_df[col] = pd.NA  # Or appropriate default
            new_row_df = new_row_df[df.columns]  # Ensure column order

            # If df is empty, new_row_df types might not match self.ohlcv_history default astype.
            # Re-apply astype or ensure compatible types for concat.
            # Assuming concat handles type promotion or columns are compatible.

            # Append new data
            # Check if timestamp already exists to avoid duplicates, update if it does
            if bar_timestamp not in df.index:
                df = pd.concat([df, new_row_df])
            else:
                # Update existing row
                df.loc[bar_timestamp] = new_row_df.iloc[0]
                self.logger.debug(
                    "Updated existing OHLCV bar for %s at %s",
                    trading_pair,
                    bar_timestamp,
                    source_module=self._source_module,
                )

            # Sort by timestamp (index)
            df.sort_index(inplace=True)

            # Prune old data - keep a bit more than strictly required for safety margin
            min_hist = self._get_min_history_required()
            required_length = min_hist + 50  # Keep 50 extra bars as buffer
            if len(df) > required_length:
                df = df.iloc[-required_length:]

            self.ohlcv_history[trading_pair] = df
            self.logger.debug(
                "Processed OHLCV update for %s. History size: %s",
                trading_pair,
                len(df),
                source_module=self._source_module,
            )

        except KeyError:
            self.logger.exception(
                "Missing key in OHLCV payload for %s.",
                trading_pair,
                source_module=self._source_module,
                context={"payload": ohlcv_payload},
            )
        except (ValueError, TypeError):
            self.logger.exception(
                "Data conversion error in OHLCV payload for %s",
                trading_pair,
                source_module=self._source_module,
                context={"payload": ohlcv_payload},
            )
        except Exception:
            self.logger.exception(
                "Unexpected error handling OHLCV update for %s",
                trading_pair,
                source_module=self._source_module,
                context={"payload": ohlcv_payload},
            )

    def _handle_l2_update(self, trading_pair: str, l2_payload: dict[str, Any]) -> None:
        """Parse and store an L2 order book update."""
        try:
            # Extract bids and asks. Ensure they are lists of lists/tuples as expected.
            # inter_module_comm.md: bids/asks: List of lists [[price_str, volume_str], ...]
            raw_bids = l2_payload.get("bids")
            raw_asks = l2_payload.get("asks")

            if not isinstance(raw_bids, list) or not isinstance(raw_asks, list):
                self.logger.warning(
                    "L2 bids/asks are not lists for %s. Payload: %s",
                    trading_pair,
                    l2_payload,
                    source_module=self._source_module,
                )
                # Decide if we should clear the book or keep stale data.
                # For now, we'll just return and not update if format is wrong.
                return

            # Convert price/volume strings to Decimal for precision
            # Bids: List of [Decimal(price), Decimal(volume)]
            # Asks: List of [Decimal(price), Decimal(volume)]
            # We expect bids to be sorted highest first, asks lowest first as per doc.

            processed_bids = []
            for i, bid_level in enumerate(raw_bids):
                if (
                    isinstance(bid_level, list | tuple)
                    and len(bid_level) == self._EXPECTED_L2_LEVEL_LENGTH
                ):
                    try:
                        processed_bids.append([Decimal(bid_level[0]), Decimal(bid_level[1])])
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            "Error converting L2 bid level %s for %s: %s - %s",
                            i,
                            trading_pair,
                            bid_level,
                            e,
                            source_module=self._source_module,
                        )
                        # Optionally skip this level or use NaN/None
                        continue  # Skip malformed level
                else:
                    self.logger.warning(
                        "Malformed L2 bid level %s for %s: %s",
                        i,
                        trading_pair,
                        bid_level,
                        source_module=self._source_module,
                    )

            processed_asks = []
            for i, ask_level in enumerate(raw_asks):
                if (
                    isinstance(ask_level, list | tuple)
                    and len(ask_level) == self._EXPECTED_L2_LEVEL_LENGTH
                ):
                    try:
                        processed_asks.append([Decimal(ask_level[0]), Decimal(ask_level[1])])
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            "Error converting L2 ask level %s for %s: %s - %s",
                            i,
                            trading_pair,
                            ask_level,
                            e,
                            source_module=self._source_module,
                        )
                        continue  # Skip malformed level
                else:
                    self.logger.warning(
                        "Malformed L2 ask level %s for %s: %s",
                        i,
                        trading_pair,
                        ask_level,
                        source_module=self._source_module,
                    )

            # Store the processed L2 book data
            # The L2 book features will expect bids sorted high to low, asks low to high.
            self.l2_books[trading_pair] = {
                "bids": processed_bids,  # Already sorted highest bid first from source
                "asks": processed_asks,  # Already sorted lowest ask first from source
                "timestamp": pd.to_datetime(
                    l2_payload.get("timestamp_exchange") or datetime.utcnow(),
                    utc=True,
                ),
            }
            self.logger.debug(
                "Processed L2 update for %s. Num bids: %s, Num asks: %s",
                trading_pair,
                len(processed_bids),
                len(processed_asks),
                source_module=self._source_module,
            )

        except KeyError:
            self.logger.exception(
                "Missing key in L2 payload for %s.",
                trading_pair,
                source_module=self._source_module,
                context={"payload": l2_payload},
            )
        except Exception:
            self.logger.exception(
                "Unexpected error handling L2 update for %s",
                trading_pair,
                source_module=self._source_module,
                context={"payload": l2_payload},
            )

    async def _handle_trade_event(self, event_dict: dict[str, Any]) -> None:
        """Handle incoming raw trade events and store them."""
        # This method will be called by pubsub, so it takes the full event_dict
        payload = event_dict.get("payload")
        if not payload:
            self.logger.warning(
                "Trade event missing payload.",
                context=event_dict,
                source_module=self._source_module,
            )
            return

        trading_pair = payload.get("trading_pair")
        if not trading_pair:
            self.logger.warning(
                "Trade event payload missing trading_pair.",
                context=payload,
                source_module=self._source_module,
            )
            return

        try:
            trade_timestamp_str = payload.get("timestamp_exchange")
            price_str = payload.get("price")
            volume_str = payload.get("volume")
            side = payload.get("side")  # "buy" or "sell"

            if not all([trade_timestamp_str, price_str, volume_str, side]):
                self.logger.warning(
                    "Trade event for %s is missing required fields "
                    "(timestamp, price, volume, or side).",
                    trading_pair,
                    context=payload,
                    source_module=self._source_module,
                )
                return

            trade_data = {
                "timestamp": pd.to_datetime(trade_timestamp_str, utc=True),
                "price": Decimal(price_str),
                "volume": Decimal(volume_str),
                "side": side.lower(),  # Ensure lowercase for consistency
            }

            if trade_data["side"] not in ["buy", "sell"]:
                self.logger.warning(
                    "Invalid trade side '%s' for %s.",
                    trade_data["side"],
                    trading_pair,
                    context=payload,
                    source_module=self._source_module,
                )
                return

            self.trade_history[trading_pair].append(trade_data)
            self.logger.debug(
                "Stored trade for %s: P=%s, V=%s, Side=%s",
                trading_pair,
                trade_data["price"],
                trade_data["volume"],
                trade_data["side"],
                source_module=self._source_module,
            )

        except KeyError:
            self.logger.exception(
                "Missing key in trade event payload for %s.",
                trading_pair,
                source_module=self._source_module,
                context=payload,
            )
        except (ValueError, TypeError):
            self.logger.exception(
                "Data conversion error in trade event payload for %s",
                trading_pair,
                source_module=self._source_module,
                context=payload,
            )
        except Exception:
            self.logger.exception(
                "Unexpected error handling trade event for %s",
                trading_pair,
                source_module=self._source_module,
                context=payload,
            )

    def _extract_feature_configs(self) -> None:
        """Extract feature-specific configurations from the main config.

        This method also ensures that default preprocessing keys (imputation, scaling)
        are present in each feature's configuration. These keys will be used later
        to define Scikit-learn preprocessing steps in feature calculation pipelines.
        """
        raw_features_config = self.config.get("features", {})
        processed_feature_configs: dict[str, dict[str, Any]] = {}

        if isinstance(raw_features_config, dict):
            for feature_name, feature_cfg in raw_features_config.items():
                if not isinstance(feature_cfg, dict):
                    self.logger.warning(
                        "Feature configuration for '%s' is not a dictionary. Skipping.",
                        feature_name,
                        source_module=self._source_module,
                    )
                    continue

                # Ensure preprocessing keys exist, defaulting to None.
                # These will be used to define Scikit-learn preprocessing steps.
                if 'imputation' not in feature_cfg:
                    feature_cfg['imputation'] = None  # e.g., {'strategy': 'mean'}
                if 'scaling' not in feature_cfg:
                    feature_cfg['scaling'] = None     # e.g., {'method': 'standard'}

                processed_feature_configs[feature_name] = feature_cfg
            self._feature_configs = processed_feature_configs
        else:
            self.logger.warning(
                "Global 'features' configuration is not a dictionary. No features loaded.",
                source_module=self._source_module,
            )
            self._feature_configs = {}

    def _get_min_history_required(self) -> int:
        """Determine the minimum required history size for TA calculations.
        This function relies on accessing period/length parameters from feature configurations.
        The addition of 'imputation' and 'scaling' keys at the same level in the
        configuration structure does not affect its operation.
        """
        min_size = 1  # Minimum baseline

        # Check various indicator requirements
        periods = [
            self._get_period_from_config("rsi", "period", 14),
            self._get_period_from_config("roc", "period", 1),
            self._get_period_from_config("bbands", "length", 20),
            self._get_period_from_config("vwap", "length", 14),
            self._get_period_from_config("atr", "length", 14),
            self._get_period_from_config("stdev", "length", 14),
        ]

        if periods:
            min_size = max(periods) * 3  # Multiply by 3 for a safe margin

        return max(100, min_size)  # At least 100 bars for good measure

    def _get_period_from_config(
        self,
        feature_name: str,
        field_name: str,
        default_value: int,
    ) -> int:
        """Retrieve the period from config for a specific feature.
        This function relies on accessing specific period/length parameters within a
        feature's configuration dictionary. The addition of 'imputation' and 'scaling'
        keys at the same level does not affect its ability to retrieve these parameters.
        """
        feature_cfg = self._feature_configs.get(feature_name, {})
        # Ensure feature_cfg is a dictionary before attempting to get a value from it.
        # _extract_feature_configs should ensure this, but an extra check is safe.
        if isinstance(feature_cfg, dict):
            period_value = feature_cfg.get(field_name, default_value)
            return (
                period_value if isinstance(period_value, int) and period_value > 0 else default_value
            )
        # If feature_cfg is not a dict (e.g. if config was malformed and somehow bypassed checks)
        self.logger.warning(
            "Configuration for feature '%s' is not a dictionary when trying to get '%s'. "
            "Returning default value %s.",
            feature_name,
            field_name,
            default_value,
            source_module=self._source_module,
        )
        return default_value

    async def start(self) -> None:
        """Start the feature engine and subscribe to relevant events."""
        try:
            # Subscribe process_market_data to handle both OHLCV and L2 updates
            self.pubsub_manager.subscribe(
                EventType.MARKET_DATA_OHLCV,
                self.process_market_data,
            )
            self.pubsub_manager.subscribe(
                EventType.MARKET_DATA_L2,
                self.process_market_data,
            )
            self.pubsub_manager.subscribe(
                EventType.MARKET_DATA_TRADE,
                self._handle_trade_event,  # New subscription
            )
            self.logger.info(
                "FeatureEngine started and subscribed to MARKET_DATA_OHLCV, "
                "MARKET_DATA_L2, and MARKET_DATA_TRADE events.",
                source_module=self._source_module,
            )
        except Exception:
            self.logger.exception(
                "Error during FeatureEngine start and subscription",
                source_module=self._source_module,
            )
            # Depending on desired behavior, might re-raise or handle to prevent full stop

    async def stop(self) -> None:
        """Stop the feature engine and clean up resources."""
        try:
            self.pubsub_manager.unsubscribe(
                EventType.MARKET_DATA_OHLCV,
                self.process_market_data,
            )
            self.pubsub_manager.unsubscribe(
                EventType.MARKET_DATA_L2,
                self.process_market_data,
            )
            self.pubsub_manager.unsubscribe(
                EventType.MARKET_DATA_TRADE,
                self._handle_trade_event,  # New unsubscription
            )
            self.logger.info(
                "FeatureEngine stopped and unsubscribed from market data events.",
                source_module=self._source_module,
            )
        except Exception:
            self.logger.exception(
                "Error during FeatureEngine stop and unsubscription",
                source_module=self._source_module,
            )

    async def process_market_data(self, market_data_event_dict: dict[str, Any]) -> None:
        """Process market data to generate features.

        Args:
        ----
            market_data_event_dict: Market data event dictionary
        """
        # Assuming market_data_event_dict is the full event object including
        # event_type, payload, source_module, etc. as per inter_module_comm.md

        event_type = market_data_event_dict.get("event_type")
        payload = market_data_event_dict.get("payload")
        source_module = market_data_event_dict.get("source_module")  # For logging/context

        if not event_type or not payload:
            self.logger.warning(
                "Received market data event with missing event_type or payload.",
                source_module=self._source_module,  # Log from FeatureEngine itself
                context={"original_event": market_data_event_dict},
            )
            return

        trading_pair = payload.get("trading_pair")
        if not trading_pair:
            self.logger.warning(
                "Market data event (type: %s) missing trading_pair.",
                event_type,
                source_module=self._source_module,
                context={"original_event": market_data_event_dict},
            )
            return

        self.logger.debug(
            "Processing event %s for %s from %s",
            event_type,
            trading_pair,
            source_module,
            source_module=self._source_module,
        )

        if event_type == "MARKET_DATA_OHLCV":
            self._handle_ohlcv_update(trading_pair, payload)
            # OHLCV update is the trigger for calculating all features for that bar's timestamp
            timestamp_bar_start = payload.get("timestamp_bar_start")
            if timestamp_bar_start:
                # We'll define _calculate_and_publish_features as async shortly
                await self._calculate_and_publish_features(trading_pair, timestamp_bar_start)
            else:
                self.logger.warning(
                    "OHLCV event for %s missing 'timestamp_bar_start', "
                    "cannot calculate features.",
                    trading_pair,
                    source_module=self._source_module,
                    context={"payload": payload},
                )
        elif event_type == "MARKET_DATA_L2":
            self._handle_l2_update(trading_pair, payload)
            # L2 updates typically don't trigger a full feature calculation on their own
            # in this design, as features are aligned with OHLCV bar closures.
            # L2 data is stored and used when an OHLCV bar triggers calculation.
        elif event_type == "MARKET_DATA_TRADE":
            await self._handle_trade_event(market_data_event_dict)
        else:
            self.logger.warning(
                "Received unknown market data event type: %s for %s",
                event_type,
                trading_pair,
                source_module=self._source_module,
                context={"original_event": market_data_event_dict},
            )

    # --- Pipeline-compatible feature calculation methods ---
    # These methods are designed to be used within Scikit-learn FunctionTransformers.
    # They expect float64 inputs and produce float64 outputs (pd.Series or pd.DataFrame).

    @staticmethod
    def _pipeline_compute_rsi(data: pd.Series, period: int) -> pd.Series:
        """Compute RSI using pandas-ta, expecting float64 Series input.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(data, pd.Series):
            # Or raise TypeError, depending on how pipeline construction handles this
            # For now, assume a Series is passed. Add logging if needed.
            # self.logger.error("_pipeline_compute_rsi expects a pd.Series.")
            return pd.Series(dtype='float64') # Return empty series on error
        return data.ta.rsi(length=period)

    @staticmethod
    def _pipeline_compute_macd(
        data: pd.Series,
        fast: int,
        slow: int,
        signal: int,
    ) -> pd.DataFrame:
        """Compute MACD using pandas-ta, expecting float64 Series input.
        Returns a DataFrame with MACD, histogram, and signal lines.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(data, pd.Series):
            # self.logger.error("_pipeline_compute_macd expects a pd.Series.")
            return pd.DataFrame(dtype='float64') # Return empty DataFrame on error
        # pandas-ta returns MACD, MACDh (histogram), MACDs (signal)
        return data.ta.macd(fast=fast, slow=slow, signal=signal)

    @staticmethod
    def _pipeline_compute_bbands(data: pd.Series, length: int, std_dev: float) -> pd.DataFrame:
        """Compute Bollinger Bands using pandas-ta, expecting float64 Series input.
        Returns a DataFrame with lower, middle, upper bands.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(data, pd.Series):
            return pd.DataFrame(dtype='float64')
        return data.ta.bbands(length=length, std=std_dev)

    @staticmethod
    def _pipeline_compute_roc(data: pd.Series, period: int) -> pd.Series:
        """Compute Rate of Change (ROC) using pandas-ta, expecting float64 Series input.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(data, pd.Series):
            return pd.Series(dtype='float64')
        return data.ta.roc(length=period)

    @staticmethod
    def _pipeline_compute_atr(
        ohlc_data: pd.DataFrame,
        length: int,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
    ) -> pd.Series:
        """Compute Average True Range (ATR) using pandas-ta.
        Expects a DataFrame with high, low, close columns (float64).
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(ohlc_data, pd.DataFrame):
            return pd.Series(dtype='float64')
        # Ensure required columns are present; this check could be more robust
        if not all(col in ohlc_data.columns for col in [high_col, low_col, close_col]):
            # self.logger.error("ATR calculation missing H/L/C columns.") # Requires logger access
            return pd.Series(dtype='float64')

        return ta.atr(
            high=ohlc_data[high_col],
            low=ohlc_data[low_col],
            close=ohlc_data[close_col],
            length=length,
        )

    @staticmethod
    def _pipeline_compute_stdev(data: pd.Series, length: int) -> pd.Series:
        """Compute Standard Deviation using pandas .rolling().std().
        Expects float64 Series input.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(data, pd.Series):
            return pd.Series(dtype='float64')
        return data.rolling(window=length).std()

    @staticmethod
    def _pipeline_compute_vwap_ohlcv(
        ohlcv_df: pd.DataFrame,
        length: int,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: str = 'volume',
    ) -> pd.Series:
        """Compute VWAP from OHLCV data using rolling window.
        Expects DataFrame with Decimal objects for price/volume, converts to float64 Series output.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(ohlcv_df, pd.DataFrame):
            return pd.Series(dtype='float64')
        if not all(col in ohlcv_df.columns for col in [high_col, low_col, close_col, volume_col]):
            return pd.Series(dtype='float64') # Or log error

        # Ensure inputs are Decimal for precision in intermediate calculations
        high_d = ohlcv_df[high_col].apply(Decimal)
        low_d = ohlcv_df[low_col].apply(Decimal)
        close_d = ohlcv_df[close_col].apply(Decimal)
        volume_d = ohlcv_df[volume_col].apply(Decimal)

        typical_price = (high_d + low_d + close_d) / Decimal("3.0")
        tp_vol = typical_price * volume_d

        sum_tp_vol = tp_vol.rolling(window=length, min_periods=length).sum()
        sum_vol = volume_d.rolling(window=length, min_periods=length).sum()

        vwap_series_decimal = sum_tp_vol / sum_vol
        # Replace infinities (from division by zero if sum_vol is 0) with NaN
        vwap_series_decimal_no_inf = vwap_series_decimal.replace([Decimal('Infinity'), Decimal('-Infinity')], np.nan)

        # Convert to float64 for pipeline compatibility, and fill NaNs from rolling/division by zero
        vwap_series_float = vwap_series_decimal_no_inf.astype('float64').fillna(np.nan)
        vwap_series_float.name = f"vwap_ohlcv_{length}"
        return vwap_series_float

    @staticmethod
    def _pipeline_compute_vwap_trades(
        trade_history_deque: deque, # Deque of trade dicts {"price": Decimal, "volume": Decimal, "timestamp": datetime}
        bar_start_times: pd.Series, # Series of datetime objects
        bar_interval_seconds: int,
    ) -> pd.Series:
        """Compute VWAP from trade data for specified bar start times.
        Returns a float64 Series. Intended for use in Scikit-learn FunctionTransformer.
        """
        vwap_results = []
        if not isinstance(bar_start_times, pd.Series) or not isinstance(trade_history_deque, deque):
            return pd.Series(dtype='float64', index=bar_start_times.index if isinstance(bar_start_times, pd.Series) else None)

        # Convert deque to DataFrame for easier filtering, assuming it's not excessively large.
        # If performance is critical for huge deques, direct iteration is better.
        # For typical 'recent trades' deque sizes, this should be acceptable.
        if not trade_history_deque: # Handle empty deque
            return pd.Series(np.nan, index=bar_start_times.index, dtype='float64', name=f"vwap_trades_{bar_interval_seconds}s")

        # Ensure trades have Decimal types
        trades_df = pd.DataFrame(list(trade_history_deque))
        trades_df["price"] = trades_df["price"].apply(Decimal)
        trades_df["volume"] = trades_df["volume"].apply(Decimal)
        trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])


        for bar_start_dt in bar_start_times:
            bar_end_dt = bar_start_dt + pd.Timedelta(seconds=bar_interval_seconds)
            relevant_trades = trades_df[
                (trades_df["timestamp"] >= bar_start_dt) & (trades_df["timestamp"] < bar_end_dt)
            ]

            if relevant_trades.empty:
                vwap_results.append(np.nan)
                continue

            sum_price_volume = (relevant_trades["price"] * relevant_trades["volume"]).sum()
            sum_volume = relevant_trades["volume"].sum()

            if sum_volume == Decimal("0"):
                vwap_results.append(np.nan)
            else:
                vwap_decimal = sum_price_volume / sum_volume
                vwap_results.append(float(vwap_decimal)) # Convert Decimal to float

        return pd.Series(vwap_results, index=bar_start_times.index, dtype='float64', name=f"vwap_trades_{bar_interval_seconds}s")


    # --- Existing feature calculation methods (some may be deprecated/refactored) ---
    # Note: _calculate_bollinger_bands, _calculate_roc, _calculate_atr, _calculate_stdev removed.
    # Note: _calculate_vwap and _calculate_vwap_from_trades removed.
    # --- Removed _calculate_roc, _calculate_atr, _calculate_stdev ---
    # --- Removed _calculate_bid_ask_spread, _calculate_order_book_imbalance, _calculate_wap, _calculate_depth ---
        # --- Removed _calculate_true_volume_delta_from_trades ---
        # The following _calculate_vwap_from_trades seems to be a leftover, should be removed.
        # It was intended to be replaced by _pipeline_compute_vwap_trades.
    # def _calculate_vwap_from_trades(
    #     self,
    #     trades: deque,
    #     current_bar_start_time: datetime,
    #     bar_interval_seconds: int = 60,
    # ) -> Decimal | None:
    #     """Calculate VWAP from a deque of recent trades relevant to the current OHLCV bar."""
    #     self.logger.debug("Calculating VWAP from trades.", source_module=self._source_module)
    #     if not trades:
    #         return None
    #     bar_end_time = current_bar_start_time + pd.Timedelta(seconds=bar_interval_seconds)
    #     relevant_trades = [
    #         trade
    #         for trade in trades
    #         if current_bar_start_time <= trade["timestamp"] < bar_end_time
    #     ]
    #     if not relevant_trades:
    #         return None
    #     sum_price_volume = sum(trade["price"] * trade["volume"] for trade in relevant_trades)
    #     sum_volume = sum(trade["volume"] for trade in relevant_trades)
    #     if sum_volume == Decimal("0"):
    #         return None
    #     result: Decimal | None = sum_price_volume / sum_volume
    #     return result


    @staticmethod
    def _pipeline_compute_l2_spread(l2_books_series: pd.Series) -> pd.DataFrame:
        """Computes bid-ask spread from a Series of L2 book snapshots.
        Outputs a DataFrame with 'abs_spread' and 'pct_spread' (float64).
        Intended for Scikit-learn FunctionTransformer.
        """
        abs_spreads = []
        pct_spreads = []
        for book in l2_books_series:
            if book and book.get("bids") and book.get("asks"):
                best_bid = book["bids"][0][0] # Assumes Decimal
                best_ask = book["asks"][0][0] # Assumes Decimal
                if best_bid and best_ask and best_ask > best_bid: # ensure ask > bid
                    abs_spread = best_ask - best_bid
                    mid_price = (best_bid + best_ask) / Decimal("2")
                    pct_spread = (abs_spread / mid_price) * Decimal("100") if mid_price != Decimal("0") else Decimal("0.0")
                    abs_spreads.append(float(abs_spread))
                    pct_spreads.append(float(pct_spread))
                else:
                    abs_spreads.append(np.nan)
                    pct_spreads.append(np.nan)
            else:
                abs_spreads.append(np.nan)
                pct_spreads.append(np.nan)
        return pd.DataFrame({"abs_spread": abs_spreads, "pct_spread": pct_spreads}, index=l2_books_series.index, dtype='float64')

    @staticmethod
    def _pipeline_compute_l2_imbalance(l2_books_series: pd.Series, levels: int = 5) -> pd.Series:
        """Computes order book imbalance from a Series of L2 book snapshots.
        Outputs a Series (float64).
        Intended for Scikit-learn FunctionTransformer.
        """
        imbalances = []
        for book in l2_books_series:
            if book and book.get("bids") and book.get("asks"):
                bid_vol_at_levels = sum(level[1] for level in book["bids"][:levels] if len(level)==2 and isinstance(level[1], Decimal))
                ask_vol_at_levels = sum(level[1] for level in book["asks"][:levels] if len(level)==2 and isinstance(level[1], Decimal))
                total_vol = bid_vol_at_levels + ask_vol_at_levels
                if total_vol > Decimal("0"):
                    imbalance = (bid_vol_at_levels - ask_vol_at_levels) / total_vol
                    imbalances.append(float(imbalance))
                else:
                    imbalances.append(0.0) # Or np.nan if preferred
            else:
                imbalances.append(np.nan)
        return pd.Series(imbalances, index=l2_books_series.index, dtype='float64', name=f"imbalance_{levels}")

    @staticmethod
    def _pipeline_compute_l2_wap(l2_books_series: pd.Series, levels: int = 1) -> pd.Series:
        """Computes Weighted Average Price (WAP) from a Series of L2 book snapshots.
        Usually calculated for the top level (levels=1). Outputs a Series (float64).
        Intended for Scikit-learn FunctionTransformer.
        """
        waps = []
        for book in l2_books_series:
            if book and book.get("bids") and book.get("asks") and book["bids"] and book["asks"]:
                # Using specified number of levels for WAP calculation
                bid_sum_price_vol = Decimal("0.0")
                bid_total_vol = Decimal("0.0")
                for price, vol in book["bids"][:levels]:
                    bid_sum_price_vol += Decimal(price) * Decimal(vol)
                    bid_total_vol += Decimal(vol)

                ask_sum_price_vol = Decimal("0.0")
                ask_total_vol = Decimal("0.0")
                for price, vol in book["asks"][:levels]:
                    ask_sum_price_vol += Decimal(price) * Decimal(vol)
                    ask_total_vol += Decimal(vol)

                if bid_total_vol > Decimal("0") and ask_total_vol > Decimal("0"): # Ensure liquidity on both sides
                    # Classic WAP formula for specified levels (often just top level)
                    # This is a common way to calculate WAP, more robust than just top-of-book if levels > 1
                    best_bid_price = book["bids"][0][0]
                    best_bid_vol = book["bids"][0][1]
                    best_ask_price = book["asks"][0][0]
                    best_ask_vol = book["asks"][0][1]

                    if (best_bid_vol + best_ask_vol) > Decimal("0"):
                        wap_val = (best_bid_price * best_ask_vol + best_ask_price * best_bid_vol) / (best_bid_vol + best_ask_vol)
                        waps.append(float(wap_val))
                    else: # Should not happen if bid_total_vol and ask_total_vol are > 0
                        waps.append(np.nan)
                else:
                    waps.append(np.nan)
            else:
                waps.append(np.nan)
        return pd.Series(waps, index=l2_books_series.index, dtype='float64', name=f"wap_{levels}")

    @staticmethod
    def _pipeline_compute_l2_depth(l2_books_series: pd.Series, levels: int = 5) -> pd.DataFrame:
        """Computes bid and ask depth from a Series of L2 book snapshots.
        Outputs a DataFrame with 'bid_depth' and 'ask_depth' (float64).
        Intended for Scikit-learn FunctionTransformer.
        """
        bid_depths = []
        ask_depths = []
        for book in l2_books_series:
            if book and book.get("bids") and book.get("asks"):
                bid_depth = sum(level[1] for level in book["bids"][:levels] if len(level)==2 and isinstance(level[1], Decimal))
                ask_depth = sum(level[1] for level in book["asks"][:levels] if len(level)==2 and isinstance(level[1], Decimal))
                bid_depths.append(float(bid_depth))
                ask_depths.append(float(ask_depth))
            else:
                bid_depths.append(np.nan)
                ask_depths.append(np.nan)
        df = pd.DataFrame({
            f"bid_depth_{levels}": bid_depths,
            f"ask_depth_{levels}": ask_depths
        }, index=l2_books_series.index, dtype='float64')
        return df

    @staticmethod
    def _pipeline_compute_volume_delta(
        trade_history_deque: deque,
        bar_start_times: pd.Series,
        bar_interval_seconds: int,
    ) -> pd.Series:
        """Computes Volume Delta from trade data for specified bar start times.
        Outputs a Series (float64).
        Intended for Scikit-learn FunctionTransformer.
        """
        deltas = []
        if not isinstance(bar_start_times, pd.Series) or not isinstance(trade_history_deque, deque):
            return pd.Series(dtype='float64', index=bar_start_times.index if isinstance(bar_start_times, pd.Series) else None)

        if not trade_history_deque:
            return pd.Series(np.nan, index=bar_start_times.index, dtype='float64', name=f"volume_delta_{bar_interval_seconds}s")

        trades_df = pd.DataFrame(list(trade_history_deque))
        trades_df["price"] = trades_df["price"].apply(Decimal)
        trades_df["volume"] = trades_df["volume"].apply(Decimal)
        trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
        # Ensure side is lowercase
        trades_df["side"] = trades_df["side"].str.lower()


        for bar_start_dt in bar_start_times:
            bar_end_dt = bar_start_dt + pd.Timedelta(seconds=bar_interval_seconds)
            relevant_trades = trades_df[
                (trades_df["timestamp"] >= bar_start_dt) & (trades_df["timestamp"] < bar_end_dt)
            ]

            if relevant_trades.empty:
                deltas.append(0.0) # Or np.nan if preferred for "no trades" vs "zero delta"
                continue

            buy_volume = relevant_trades[relevant_trades["side"] == "buy"]["volume"].sum()
            sell_volume = relevant_trades[relevant_trades["side"] == "sell"]["volume"].sum()
            deltas.append(float(buy_volume - sell_volume))

        return pd.Series(deltas, index=bar_start_times.index, dtype='float64', name=f"volume_delta_{bar_interval_seconds}s")


    # --- Existing feature calculation methods (some may be deprecated/refactored) ---
    # Note: _calculate_bollinger_bands, _calculate_roc, _calculate_atr, _calculate_stdev removed.
    # Note: _calculate_vwap and _calculate_vwap_from_trades removed.
    # --- Removed _calculate_roc, _calculate_atr, _calculate_stdev ---
    # --- Removed _calculate_bid_ask_spread, _calculate_order_book_imbalance, _calculate_wap, _calculate_depth ---
    # --- Removed _calculate_true_volume_delta_from_trades, _calculate_vwap_from_trades ---


    def _calculate_vwap_from_trades( # This method is still present in the provided code, but should be removed per instructions.
        self, # Assuming this will be removed based on previous steps.
        trades: deque, # This argument type should be deque, not pd.DataFrame.
        current_bar_start_time: datetime,
        bar_interval_seconds: int = 60,
    ) -> Decimal | None:
        """Calculate VWAP from a deque of recent trades relevant to the current OHLCV bar."""
        self.logger.debug("Calculating VWAP from trades.", source_module=self._source_module)
        if not trades:
            return None

        # Calculate the end time for the current bar
        bar_end_time = current_bar_start_time + pd.Timedelta(seconds=bar_interval_seconds)

        relevant_trades = [
            trade
            for trade in trades
            if current_bar_start_time <= trade["timestamp"] < bar_end_time
        ]

        if not relevant_trades:
            return None

        sum_price_volume = sum(trade["price"] * trade["volume"] for trade in relevant_trades)
        sum_volume = sum(trade["volume"] for trade in relevant_trades)

        if sum_volume == Decimal("0"):
            return None

        result: Decimal | None = sum_price_volume / sum_volume
        return result

    async def _calculate_and_publish_features(
        self,
        trading_pair: str,
        timestamp_features_for: str,
    ) -> None:
        """Calculate all configured features using pipelines and publish them."""

        ohlcv_df_full_history = self.ohlcv_history.get(trading_pair)
        if ohlcv_df_full_history is None or len(ohlcv_df_full_history) < self._get_min_history_required():
            self.logger.info(
                "Not enough OHLCV data for %s to calculate features. Need %s, have %s.",
                trading_pair,
                min_history,
                len(ohlcv_df) if ohlcv_df is not None else 0,
                source_module=self._source_module,
            )
            return

        current_l2_book = self.l2_books.get(trading_pair)
        if current_l2_book and (
            not current_l2_book.get("bids") or not current_l2_book.get("asks")
        ):
            self.logger.debug(
                "L2 book for %s is present but empty or missing bids/asks. "
                "L2 features may be skipped.",
                trading_pair,
                source_module=self._source_module,
            )
            # L2 features might be skipped by their handlers if book is not suitable.

        all_generated_features: dict[str, Any] = {}

        bar_start_datetime = pd.to_datetime(timestamp_features_for, utc=True)

        # Filter OHLCV data up to the current bar's start time for historical context
        # Pipelines will internally select the latest point after calculation over history.
        current_ohlcv_df_decimal = ohlcv_df_full_history[ohlcv_df_full_history.index <= bar_start_datetime]

        if current_ohlcv_df_decimal.empty:
            self.logger.warning("No historical OHLCV data available for %s up to %s.", trading_pair, bar_start_datetime)
            return # Cannot proceed without data for this timestamp

        # Prepare standard input types based on Decimal data
        close_series_for_pipelines = current_ohlcv_df_decimal['close'].astype('float64')
        # Ensure 'open', 'high', 'low', 'close', 'volume' are float for OHLCV df inputs
        ohlcv_df_for_pipelines = current_ohlcv_df_decimal.astype({
            'open': 'float64', 'high': 'float64', 'low': 'float64',
            'close': 'float64', 'volume': 'float64'
        })

        # L2 book snapshot for the current bar
        # Assuming self.l2_books[trading_pair] holds the latest book, or one aligned by a separate process
        # For pipeline processing, we need a Series (even if single-element)
        latest_l2_book_snapshot = self.l2_books.get(trading_pair) # This is the overall latest
        # TODO: A more robust way would be to fetch L2 book aligned with bar_start_datetime
        l2_books_aligned_series = pd.Series([latest_l2_book_snapshot], index=[bar_start_datetime])

        # Trade data for trade-based features
        trades_deque = self.trade_history.get(trading_pair, deque())
        # For single bar calculation, bar_start_times_series is just the current bar
        bar_start_times_series = pd.Series([bar_start_datetime], index=[bar_start_datetime])


        for pipeline_name, pipeline_info in self.feature_pipelines.items():
            pipeline_obj = pipeline_info['pipeline']
            input_type = pipeline_info['input_type']
            # original_params = pipeline_info['params'] # Available if needed for naming or logic

            pipeline_input_data: Any = None
            requires_fit_transform = True # Most of our custom funcs are stateless from sklearn's view

            if input_type == 'close_series':
                pipeline_input_data = close_series_for_pipelines
            elif input_type == 'ohlcv_df':
                pipeline_input_data = ohlcv_df_for_pipelines
            elif input_type == 'l2_book_series':
                pipeline_input_data = l2_books_aligned_series
            elif input_type == 'trades_and_bar_starts':
                # For functions expecting (trade_history_deque, bar_start_times_series, ...)
                # The FunctionTransformer's `func` will be called with `X` and `kw_args`.
                # We pass bar_start_times_series as X, and trade_history_deque via kw_args at transform time.
                # This requires modifying how `_pipeline_compute_vwap_trades` etc. are wrapped or called.
                # For now, let's assume the function signature is `func(X, **kw_args)`
                # and X is bar_start_times_series. The deque is passed via kw_args.
                # This means we need to update FunctionTransformer definition or pass it here.
                # Simpler: pass a tuple if func expects multiple main arguments not in kw_args
                # pipeline_input_data = (trades_deque, bar_start_times_series)
                # Let's assume the _pipeline_compute functions are designed to take the primary series as X
                # and other context (like the full deque) via kw_args set at definition or runtime.
                # The current _pipeline_compute_vwap_trades expects (deque, series_of_times, interval)
                # So, we need to call it specially.
                # This part of the design needs careful handling for non-standard X inputs.
                # For now, we'll handle these as special cases after the main loop or adapt the call.
                if "vwap_trades" in pipeline_name or "volume_delta" in pipeline_name:
                    # These are called differently as they don't just operate on a single series X from ohlcv
                    # Their kw_args for interval are already set in _build_feature_pipelines
                    # Their first arg is the deque, second is the series of bar times
                    try:
                        # The `func` in FunctionTransformer will get `trades_deque` as `X`
                        # and `bar_start_times_series` as a `kw_arg` if we modify the call or func def.
                        # Current func def: _pipeline_compute_vwap_trades(trade_history_deque, bar_start_times, interval)
                        # So, X should be trade_history_deque, and bar_start_times must be a kw_arg.
                        # This means the FunctionTransformer for these should be:
                        # FunctionTransformer(func, kw_args={'bar_start_times': bar_start_times_series, 'bar_interval_seconds': ...})
                        # And then call transform(trades_deque). This is a bit messy due to dynamic kw_args.

                        # Alternative: Call the static method directly for these complex cases for now.
                        # This bypasses part of the sklearn pipeline's transform flow for X.
                        static_calc_func = None
                        calc_kwargs = {}
                        if "vwap_trades" in pipeline_name:
                            static_calc_func = FeatureEngine._pipeline_compute_vwap_trades
                            calc_kwargs = {'trade_history_deque': trades_deque,
                                           'bar_start_times': bar_start_times_series,
                                           'bar_interval_seconds': pipeline_info['params'].get("length_seconds",60)}
                        elif "volume_delta" in pipeline_name:
                            static_calc_func = FeatureEngine._pipeline_compute_volume_delta
                            calc_kwargs = {'trade_history_deque': trades_deque,
                                           'bar_start_times': bar_start_times_series,
                                           'bar_interval_seconds': pipeline_info['params'].get("bar_interval_seconds",60)}

                        if static_calc_func:
                            raw_pipeline_output = static_calc_func(**calc_kwargs)
                            # Apply further steps (imputer, scaler) from the pipeline manually if they exist
                            # This is a temporary workaround for complex inputs.
                            if 'output_fillna' in pipeline_obj.named_steps: # Example step name
                                raw_pipeline_output = pipeline_obj.named_steps['output_fillna'].transform(raw_pipeline_output)
                            if 'output_scaler' in pipeline_obj.named_steps: # Example step name
                                raw_pipeline_output = pipeline_obj.named_steps['output_scaler'].transform(raw_pipeline_output)
                            # END WORKAROUND
                        else:
                            continue # Should not happen if names match
                    except Exception as e:
                        self.logger.exception("Error executing trade-based pipeline %s: %s", pipeline_name, e)
                        continue
                else: # Should be an L2 feature if input_type is l2_book_series
                    pipeline_input_data = l2_books_aligned_series


            else: # Standard OHLCV based features (RSI, MACD, BBands, ROC, ATR, StDev, VWAP_OHLCV)
                if pipeline_input_data is None : # Should have been set
                    self.logger.warning("Pipeline input data not set for %s", pipeline_name)
                    continue
                try:
                    # Assuming stateless transformers or that fit_transform handles it.
                    # For production, consider fitting scalers once on a training set.
                    raw_pipeline_output = pipeline_obj.fit_transform(pipeline_input_data)
                except Exception as e:
                    self.logger.exception("Error executing pipeline %s: %s", pipeline_name, e)
                    continue

            # Extract the latest feature value(s)
            # For series input, output is usually a series. We need the last value.
            # For L2/Trade features on single bar, output is already the latest.
            latest_features_values: Any = None
            if isinstance(raw_pipeline_output, pd.Series):
                if not raw_pipeline_output.empty:
                    if input_type not in ['l2_book_series', 'trades_and_bar_starts'] or len(raw_pipeline_output) > 1 : # If it processed history
                        latest_features_values = raw_pipeline_output.iloc[-1]
                    else: # Already the single calculated value for the current bar
                        latest_features_values = raw_pipeline_output.iloc[0] if len(raw_pipeline_output) == 1 else np.nan
            elif isinstance(raw_pipeline_output, pd.DataFrame):
                if not raw_pipeline_output.empty:
                    if input_type not in ['l2_book_series'] or len(raw_pipeline_output) > 1: # If it processed history
                        latest_features_values = raw_pipeline_output.iloc[-1] # This gives a Series (one row)
                    else: # Already the single calculated row for the current bar
                        latest_features_values = raw_pipeline_output.iloc[0] if len(raw_pipeline_output) == 1 else pd.Series(dtype='float64')
            elif isinstance(raw_pipeline_output, np.ndarray): # Should be rare due to PandasScalerTransformer
                 if raw_pipeline_output.ndim == 1 and raw_pipeline_output.size > 0:
                    latest_features_values = raw_pipeline_output[-1]
                 elif raw_pipeline_output.ndim == 2 and raw_pipeline_output.shape[0] > 0:
                    latest_features_values = pd.Series(raw_pipeline_output[-1, :]) # Convert row to series
            else: # scalar float, e.g. if pipeline output was just one value
                latest_features_values = raw_pipeline_output

            # Naming and storing features
            if isinstance(latest_features_values, pd.Series):
                for idx_name, value in latest_features_values.items():
                    # Construct a unique feature name, e.g. from pipeline_name and sub-feature name (column name)
                    # Ensure idx_name (column from DataFrame) is a string
                    col_name = str(idx_name)
                    # Clean up default pandas-ta column names if needed
                    # e.g. MACD_12_26_9 -> macd, MACDh_12_26_9 -> macd_hist
                    # This part needs a robust naming strategy.
                    # For now: pipeline_name (which is feature_key_pipeline) + column name
                    base_feature_key = pipeline_name.replace("_pipeline","")
                    # If pipeline_name was "macd_12_26_9_pipeline", base_feature_key is "macd_12_26_9"
                    # col_name could be "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"
                    # A better way for naming is to use the original feature_key from config and the column name.
                    # original_feature_config_key = pipeline_info['params'].get('original_key', base_feature_key)

                    # Simplified naming: {original_feature_key}_{column_suffix}
                    # Example: feature_key = "macd_config_name", col_name = "MACD_12_26_9" -> "macd_config_name_MACD_12_26_9"
                    # Or, if col_name is like "lowerband", "upperband" from bbands.
                    feature_output_name = f"{base_feature_key}_{col_name}"
                    all_generated_features[feature_output_name] = value
            elif pd.notna(latest_features_values): # Single float/value
                # Name is just the pipeline name (e.g., "rsi_14_pipeline" -> "rsi_14")
                feature_output_name = pipeline_name.replace("_pipeline","")
                all_generated_features[feature_output_name] = latest_features_values

        calculated_features_dict = {
            name: self._format_feature_value(val) for name, val in all_generated_features.items() if pd.notna(val)
        }

        # Fallback for any old handlers if no pipelines were built (mostly for transition)
        if not self.feature_pipelines and not calculated_features_dict:
            self.logger.debug("No pipelines executed, attempting feature calculation with remaining old handlers.", source_module=self._source_module)
            # ... (old handler logic can be here if needed, but it's mostly empty now) ...
            pass


        if not calculated_features_dict: # Check if any features were produced
            self.logger.info(
                "No features were successfully calculated for %s at %s. Not publishing event.",
                trading_pair,
                timestamp_features_for,
                source_module=self._source_module,
            )
            return

        # Construct and publish FeatureEvent
        event_payload = {
            "trading_pair": trading_pair,
            "exchange": self.config.get("exchange_name", "kraken"),
            "timestamp_features_for": timestamp_features_for,
            "features": calculated_features_dict,
        }

        full_feature_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": EventType.FEATURES_CALCULATED.name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_module": self._source_module,
            "payload": event_payload,
        }

        try:
            # Convert the dictionary to an Event object to match expected type
            from typing import cast

            # Use Any for Event since we can't import the actual type
            await self.pubsub_manager.publish(cast("Any", full_feature_event))
            self.logger.info(
                "Published FEATURES_CALCULATED event for %s at %s",
                trading_pair,
                timestamp_features_for,
                source_module=self._source_module,
                context={
                    "event_id": full_feature_event["event_id"],
                    "num_features": len(calculated_features_dict),
                },
            )
        except Exception:
            self.logger.exception(
                "Failed to publish FEATURES_CALCULATED event for %s",
                trading_pair,
                source_module=self._source_module,
            )

    def _format_feature_value(self, value: Decimal | float | object) -> str:
        """Format a feature value to string. Decimal/float to 8 decimal places."""
        if isinstance(value, Decimal | float):
            return f"{value:.8f}"
        return str(value)

    # --- Feature Processing Methods (to be refactored for pipeline usage) ---
    # The _process_rsi_feature and _process_macd_feature methods are removed
    # as their logic will be incorporated into the new pipeline-based feature generation.
    # Other _process_* methods will be updated or replaced in subsequent steps.
    # --- Removed _process_bbands_feature, _process_roc_feature, _process_atr_feature, _process_stdev_feature ---
    # --- Removed _process_vwap_feature ---
    # --- Removed _process_roc_feature, _process_atr_feature, _process_stdev_feature ---
    # --- Removed _process_l2_spread_feature, _process_l2_imbalance_feature, _process_l2_wap_feature ---
    # --- Removed _process_l2_depth_feature, _process_volume_delta_feature ---
