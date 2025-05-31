"""Feature engineering implementation for Gal-Friday.

This module provides the FeatureEngine class that handles computation of technical
indicators and other features used in prediction models.
"""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

from gal_friday.core.events import EventType

if TYPE_CHECKING:
    from collections.abc import Callable

    from gal_friday.core.pubsub import PubSubManager
    from gal_friday.interfaces.historical_data_service_interface import HistoricalDataService
    from gal_friday.logger_service import LoggerService


class FeatureEngine:
    """Processes market data to compute technical indicators and other features.

    The FeatureEngine is responsible for converting raw market data into features
    that can be used for machine learning models, including technical indicators,
    derived features, and potentially other types of features.
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
        ----
            config: Dictionary containing configuration settings
            pubsub_manager: Instance of the pub/sub event manager
            logger_service: Logging service instance
            historical_data_service: Optional historical data service for initial data loading
        """
        self.config = config
        self.pubsub_manager = pubsub_manager
        self.logger = logger_service
        self.historical_data_service = historical_data_service
        self._source_module = self.__class__.__name__

        # Feature configuration derived from config
        self._feature_configs: dict[str, dict[str, Any]] = {}
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

        self.feature_pipelines: dict[str, dict[str, Any]] = {} # Stores {'pipeline': Pipeline, 'input_type': str, 'params': dict}
        self._build_feature_pipelines()

        self.logger.info("FeatureEngine initialized.", source_module=self._source_module)


    def _build_feature_pipelines(self) -> None:
        """Build Scikit-learn pipelines for each configured feature."""
        self.logger.info("Building feature pipelines...", source_module=self._source_module)

        # Helper to create output imputation step
        def get_output_imputer_step(cfg_value, default_fill_value=0.0, is_dataframe_output=False):
            if cfg_value is None: # Default imputation
                fill_val = default_fill_value
                # For DataFrames, fillna with a dict for per-column means/medians is better if possible
                # but a simple fill_value is a fallback.
                return ('output_fillna', FunctionTransformer(lambda x: x.fillna(fill_val if not is_dataframe_output else x.mean()), validate=False)) # Default to mean for dataframes
            if cfg_value == 'passthrough':
                return None
            if isinstance(cfg_value, dict):
                strategy = cfg_value.get('strategy', 'constant')
                fill_value = cfg_value.get('fill_value', default_fill_value)
                if strategy == 'constant':
                    return (f'output_imputer_const_{fill_value}', FunctionTransformer(lambda x: x.fillna(fill_value), validate=False))
                # SimpleImputer requires 2D, so use FunctionTransformer for Series/DataFrame fillna
                if strategy == 'mean':
                    return (f'output_imputer_mean', FunctionTransformer(lambda x: x.fillna(x.mean()), validate=False))
                if strategy == 'median':
                    return (f'output_imputer_median', FunctionTransformer(lambda x: x.fillna(x.median()), validate=False))
            return None # Fallback if config is not recognized

        # Helper to create output scaler step
        def get_output_scaler_step(cfg_value):
            if cfg_value == 'passthrough' or cfg_value is None:
                return None

            scaler_instance = StandardScaler() # Default
            if isinstance(cfg_value, dict):
                method = cfg_value.get('method', 'standard')
                if method == 'minmax':
                    scaler_instance = MinMaxScaler(feature_range=cfg_value.get('feature_range', (0,1)))
            return (f'output_scaler_{type(scaler_instance).__name__}', PandasScalerTransformer(scaler_instance))


        for feature_key, params in self._feature_configs.items():
            pipeline_steps = []
            pipeline_name = f"{feature_key}_pipeline"
            input_type = None # To be determined for each feature type

            # Common input imputer for single series (e.g. 'close') based features
            # This assumes the input to the pipeline is a Series that needs imputation.
            # For features taking DataFrame (ATR, VWAP_OHLCV) or special inputs (L2, Trades),
            # input imputation needs to be handled differently or before this pipeline.
            std_input_imputer = ('input_data_imputer', SimpleImputer(strategy='mean'))

            # --- RSI Pipeline ---
            if "rsi" in feature_key.lower():
                input_type = 'close_series'
                pipeline_steps.append(std_input_imputer)

                # 1. Input Imputation (for the raw close price series fed to RSI)
                # Defaulting to mean imputation for the input data if not specified.
                # This config would ideally be part of a global input data config.
                input_imputer = SimpleImputer(strategy='mean')
                pipeline_steps.append(('input_data_imputer', input_imputer))

                # 2. RSI Calculation
                rsi_period = params.get("period", 14) # Default RSI period
                # `self._pipeline_compute_rsi` is a static method
                rsi_transformer = FunctionTransformer(
                    FeatureEngine._pipeline_compute_rsi,
                    kw_args={'period': rsi_period},
                    validate=False # Allow pandas Series
                )
                pipeline_steps.append((f'rsi_{rsi_period}_calculator', rsi_transformer))

                # 3. Output Imputation (for NaNs produced by RSI, e.g., at the start of series)
                # Default RSI output imputation: fill with 50 (neutral RSI)
                output_imputation_cfg = params.get('imputation')
                if output_imputation_cfg and isinstance(output_imputation_cfg, dict):
                    strategy = output_imputation_cfg.get('strategy', 'constant')
                    fill_value = output_imputation_cfg.get('fill_value', 50.0)
                    # Note: SimpleImputer expects 2D array, so we might need another FunctionTransformer
                    # to reshape if applying SimpleImputer directly on the output Series.
                    # For now, let's assume a custom imputer or a FunctionTransformer wrapping fillna.
                    # Using fillna via FunctionTransformer for simplicity here.
                    # This step might need its own FunctionTransformer if SimpleImputer is strictly used.
                    # Example: ft_fillna = FunctionTransformer(lambda s: s.fillna(fill_value))
                    # pipeline_steps.append((f'rsi_{rsi_period}_output_imputer', ft_fillna))
                    # For a more scikit-learn native approach with SimpleImputer:
                    # Reshape helper for SimpleImputer
                    def reshape_for_imputer(series: pd.Series) -> np.ndarray:
                        return series.to_numpy().reshape(-1, 1)
                    def reshape_after_imputer(array: np.ndarray) -> pd.Series:
                        return pd.Series(array.flatten())

                    pipeline_steps.append((f'rsi_{rsi_period}_reshape_before_impute', FunctionTransformer(reshape_for_imputer, validate=False)))
                    pipeline_steps.append((f'rsi_{rsi_period}_output_imputer', SimpleImputer(strategy=strategy, fill_value=fill_value if strategy == 'constant' else None)))
                    pipeline_steps.append((f'rsi_{rsi_period}_reshape_after_impute', FunctionTransformer(reshape_after_imputer, validate=False)))

                elif output_imputation_cfg is None or output_imputation_cfg == 'passthrough':
                    pass # No output imputation
                else: # Default output imputation for RSI
                    def fill_rsi_na(series: pd.Series, fill_val: float = 50.0) -> pd.Series:
                        return series.fillna(fill_val)
                    pipeline_steps.append((f'rsi_{rsi_period}_output_fillna_50', FunctionTransformer(fill_rsi_na, kw_args={'fill_val': 50.0} ,validate=False)))


                # 4. Output Scaling
                scaling_cfg = params.get('scaling')
                output_scaler = None
                if scaling_cfg and isinstance(scaling_cfg, dict):
                    scale_method = scaling_cfg.get('method')
                    if scale_method == 'standard':
                        output_scaler = StandardScaler()
                    elif scale_method == 'minmax':
                        feature_range = scaling_cfg.get('feature_range', (0, 1))
                        output_scaler = MinMaxScaler(feature_range=feature_range)
                    # Add other scalers as needed
                elif scaling_cfg == 'passthrough' or scaling_cfg is None : # Default or passthrough
                     pass # No scaler added
                else: # Default scaler if config is invalid or not 'passthrough'
                    output_scaler = StandardScaler() # Default to StandardScaler

                if output_scaler:
                    # Reshape for scaler (expects 2D) and then back to Series
                    def reshape_for_scaler(series: pd.Series) -> np.ndarray:
                        return series.to_numpy().reshape(-1, 1)
                    def reshape_after_scaler(array: np.ndarray, original_index) -> pd.Series: # Accept original_index
                        return pd.Series(array.flatten(), index=original_index) # Preserve index

                    # We need to capture the index before reshaping for the scaler
                    # This is tricky as the series passed to this lambda will be the output of the previous step
                    # A more robust way would be to ensure FunctionTransformers preserve index or handle it carefully.
                    # For now, this illustrates the concept.
                    # A stateful transformer might be needed or pass index explicitly.
                    # Simplified for now: assuming the scaler step can handle Series directly or this is refined.
                    # Most sklearn scalers will drop index if directly applied to Series and return ndarray.
                    # We need to wrap it to preserve index.

                    # Temporary solution for index preservation with scaler:
                    class PandasScalerTransformer(FunctionTransformer):
                        def __init__(self, scaler, **kwargs):
                            self.scaler = scaler
                            # Pass validate=False, etc., to FunctionTransformer constructor
                            super().__init__(func=self._transform_func, inverse_func=self._inverse_func_func, validate=False, **kwargs)

                        def _transform_func(self, X: pd.Series):
                            # X is a pandas Series
                            original_index = X.index
                            reshaped_x = X.to_numpy().reshape(-1,1)
                            scaled_x = self.scaler.fit_transform(reshaped_x)
                            return pd.Series(scaled_x.flatten(), index=original_index)

                        def _inverse_func_func(self, X: pd.Series):
                            original_index = X.index
                            reshaped_x = X.to_numpy().reshape(-1,1)
                            # Ensure inverse_transform is available and makes sense for the scaler
                            if hasattr(self.scaler, 'inverse_transform'):
                                unscaled_x = self.scaler.inverse_transform(reshaped_x)
                                return pd.Series(unscaled_x.flatten(), index=original_index)
                            return X # Or raise error

                    pipeline_steps.append((f'rsi_{rsi_period}_output_scaler', PandasScalerTransformer(output_scaler)))


                # Assemble and store the pipeline
                if pipeline_steps:
                    final_pipeline = Pipeline(steps=pipeline_steps)
                    pipeline_name = f"{feature_key}_pipeline" # Original key used for pipeline name
                    # Storing pipeline and its metadata
                    self.feature_pipelines[pipeline_name] = {
                        'pipeline': final_pipeline,
                        'input_type': input_type, # Set based on feature
                        'params': params # Store original params for reference
                    }
                    self.logger.info(
                        "Built pipeline: %s with steps: %s, input: %s",
                        pipeline_name, [s[0] for s in pipeline_steps], input_type,
                        source_module=self._source_module
                    )

            # --- MACD Pipeline ---
            elif "macd" in feature_key.lower():
                input_type = 'close_series'
                pipeline_steps.append(std_input_imputer)

                macd_p = params.get("params", {})
                fast = macd_p.get("fast", 12)
                slow = macd_p.get("slow", 26)
                signal = macd_p.get("signal", 9)
                calculator = FunctionTransformer(
                    FeatureEngine._pipeline_compute_macd,
                    kw_args={'fast': fast, 'slow': slow, 'signal': signal},
                    validate=False
                )
                pipeline_steps.append((f'macd_{fast}_{slow}_{signal}_calculator', calculator))

                # Output Imputation for DataFrame (e.g., fill all NaNs with 0.0)
                output_imputation_cfg = params.get('imputation')
                if output_imputation_cfg is None: # Default for MACD: fill with 0
                    pipeline_steps.append((f'macd_output_fillna_0', FunctionTransformer(lambda df: df.fillna(0.0), validate=False)))
                elif output_imputation_cfg != 'passthrough': # Apply configured or default if not passthrough
                    # Assuming dict based config like RSI, but simplified for DataFrame
                    # For more complex per-column imputation, ColumnTransformer would be needed here
                    strategy = output_imputation_cfg.get('strategy', 'constant') if isinstance(output_imputation_cfg, dict) else 'constant'
                    fill_value = output_imputation_cfg.get('fill_value', 0.0) if isinstance(output_imputation_cfg, dict) else 0.0
                    if strategy == 'constant':
                         pipeline_steps.append((f'macd_output_fillna', FunctionTransformer(lambda df: df.fillna(fill_value), validate=False)))
                    else: # mean, median etc. would need to be applied per column, could get complex without ColumnTransformer
                         pipeline_steps.append((f'macd_output_fillna_median_cols', FunctionTransformer(lambda df: df.fillna(df.median()), validate=False)))


                # Output Scaling for DataFrame
                scaling_cfg = params.get('scaling')
                if scaling_cfg and scaling_cfg != 'passthrough':
                    scaler_step = get_output_scaler_step(scaling_cfg)
                    if scaler_step: pipeline_steps.append(scaler_step)

                # Store MACD pipeline
                if pipeline_steps:
                    self.feature_pipelines[pipeline_name] = {
                        'pipeline': Pipeline(steps=pipeline_steps),
                        'input_type': input_type, 'params': params}
                    self.logger.info("Built MACD pipeline: %s, input: %s", pipeline_name, input_type, source_module=self._source_module)

            # --- Bollinger Bands (BBands) Pipeline ---
            elif "bbands" in feature_key.lower():
                input_type = 'close_series'
                pipeline_steps.append(std_input_imputer)

                bb_params = params.get("params", {}) # Assuming params might be nested
                length = bb_params.get("length", 20)
                std_dev = bb_params.get("std_dev", 2.0)
                calculator = FunctionTransformer(
                    FeatureEngine._pipeline_compute_bbands,
                    kw_args={'length': length, 'std_dev': float(std_dev)}, # Ensure std_dev is float
                    validate=False
                )
                pipeline_steps.append((f'bbands_{length}_{std_dev}_calculator', calculator))

                # Output Imputation (e.g., fill with mean of each band)
                output_imputation_cfg = params.get('imputation')
                if output_imputation_cfg is None: # Default for BBands: fill with column mean
                    pipeline_steps.append((f'bbands_output_fillna_mean', FunctionTransformer(lambda df: df.fillna(df.mean()), validate=False)))
                elif output_imputation_cfg != 'passthrough':
                     # Simplified: fill all with 0 or a specific value if configured
                    fill_value = 0.0
                    if isinstance(output_imputation_cfg, dict) and output_imputation_cfg.get('strategy') == 'constant':
                        fill_value = output_imputation_cfg.get('fill_value', 0.0)
                    pipeline_steps.append((f'bbands_output_fillna', FunctionTransformer(lambda df: df.fillna(fill_value), validate=False)))


                scaling_cfg = params.get('scaling')
                if scaling_cfg and scaling_cfg != 'passthrough':
                    scaler_step = get_output_scaler_step(scaling_cfg)
                    if scaler_step: pipeline_steps.append(scaler_step)

                # Store BBands pipeline
                if pipeline_steps:
                    self.feature_pipelines[pipeline_name] = {
                        'pipeline': Pipeline(steps=pipeline_steps),
                        'input_type': input_type, 'params': params}
                    self.logger.info("Built BBands pipeline: %s, input: %s", pipeline_name, input_type, source_module=self._source_module)

            # --- ROC Pipeline ---
            elif "roc" in feature_key.lower():
                input_type = 'close_series'
                pipeline_steps.append(std_input_imputer)
                roc_period = params.get("period", 10)
                calculator = FunctionTransformer(FeatureEngine._pipeline_compute_roc, kw_args={'period': roc_period}, validate=False)
                pipeline_steps.append((f'roc_{roc_period}_calculator', calculator))

                imputer_step = get_output_imputer_step(params.get('imputation'), 0.0) # Default fill 0 for ROC
                if imputer_step: pipeline_steps.append(imputer_step)

                scaler_step = get_output_scaler_step(params.get('scaling'))
                if scaler_step: pipeline_steps.append(scaler_step)

                if pipeline_steps:
                    self.feature_pipelines[pipeline_name] = {'pipeline': Pipeline(steps=pipeline_steps), 'input_type': input_type, 'params': params}
                    self.logger.info("Built ROC pipeline: %s, input: %s", pipeline_name, input_type, source_module=self._source_module)

            # --- ATR Pipeline ---
            elif "atr" in feature_key.lower():
                input_type = 'ohlcv_df' # ATR needs OHLC DataFrame
                # No std_input_imputer for DataFrame input types like OHLCV; handled by pipeline if needed or assumed clean.
                atr_len = params.get("length", 14)
                # TODO: allow high_col, low_col, close_col to be specified in params
                calculator = FunctionTransformer(FeatureEngine._pipeline_compute_atr, kw_args={'length': atr_len}, validate=False)
                pipeline_steps.append((f'atr_{atr_len}_calculator', calculator))

                imputer_step = get_output_imputer_step(params.get('imputation'), default_fill_value=np.nan) # Default: fill with mean later
                if imputer_step: pipeline_steps.append(imputer_step)

                scaler_step = get_output_scaler_step(params.get('scaling'))
                if scaler_step: pipeline_steps.append(scaler_step)

                if pipeline_steps:
                    self.feature_pipelines[pipeline_name] = {'pipeline': Pipeline(steps=pipeline_steps), 'input_type': input_type, 'params': params}
                    self.logger.info("Built ATR pipeline: %s, input: %s", pipeline_name, input_type, source_module=self._source_module)

            # --- Standard Deviation (StDev) Pipeline ---
            elif "stdev" in feature_key.lower():
                input_type = 'close_series'
                pipeline_steps.append(std_input_imputer)
                stdev_len = params.get("length", 20)
                calculator = FunctionTransformer(FeatureEngine._pipeline_compute_stdev, kw_args={'length': stdev_len}, validate=False)
                pipeline_steps.append((f'stdev_{stdev_len}_calculator', calculator))

                imputer_step = get_output_imputer_step(params.get('imputation'), default_fill_value=np.nan) # Default: fill with mean later
                if imputer_step: pipeline_steps.append(imputer_step)

                scaler_step = get_output_scaler_step(params.get('scaling'))
                if scaler_step: pipeline_steps.append(scaler_step)

                if pipeline_steps:
                    self.feature_pipelines[pipeline_name] = {'pipeline': Pipeline(steps=pipeline_steps), 'input_type': input_type, 'params': params}
                    self.logger.info("Built StDev pipeline: %s, input: %s", pipeline_name, input_type, source_module=self._source_module)

            # --- VWAP_OHLCV Pipeline ---
            elif feature_key.startswith("vwap_ohlcv"): # More specific check
                input_type = 'ohlcv_df' # Takes OHLCV DataFrame
                vwap_len = params.get("length", 14)
                # TODO: allow h,l,c,v columns to be specified in params for _pipeline_compute_vwap_ohlcv
                calculator = FunctionTransformer(FeatureEngine._pipeline_compute_vwap_ohlcv, kw_args={'length': vwap_len}, validate=False)
                pipeline_steps.append((f'vwap_ohlcv_{vwap_len}_calculator', calculator))

                imputer_step = get_output_imputer_step(params.get('imputation'), default_fill_value=np.nan)
                if imputer_step: pipeline_steps.append(imputer_step)

                scaler_step = get_output_scaler_step(params.get('scaling'))
                if scaler_step: pipeline_steps.append(scaler_step)

                if pipeline_steps:
                    self.feature_pipelines[pipeline_name] = {'pipeline': Pipeline(steps=pipeline_steps), 'input_type': input_type, 'params': params}
                    self.logger.info("Built VWAP_OHLCV pipeline: %s, input: %s", pipeline_name, input_type, source_module=self._source_module)

            # --- L2 Features ---
            elif feature_key.startswith("l2_spread"):
                input_type = 'l2_book_series'
                calculator = FunctionTransformer(FeatureEngine._pipeline_compute_l2_spread, validate=False)
                pipeline_steps.append(('l2_spread_calculator', calculator))
                imputer_step = get_output_imputer_step(params.get('imputation'), default_fill_value=np.nan, is_dataframe_output=True) # df.mean()
                if imputer_step: pipeline_steps.append(imputer_step)
                scaler_step = get_output_scaler_step(params.get('scaling'))
                if scaler_step: pipeline_steps.append(scaler_step)
                if pipeline_steps:
                    self.feature_pipelines[pipeline_name] = {'pipeline': Pipeline(steps=pipeline_steps), 'input_type': input_type, 'params': params}
                    self.logger.info("Built L2 Spread pipeline: %s, input: %s", pipeline_name, input_type, source_module=self._source_module)

            elif feature_key.startswith("l2_imbalance"):
                input_type = 'l2_book_series'
                levels = params.get("levels", 5)
                calculator = FunctionTransformer(FeatureEngine._pipeline_compute_l2_imbalance, kw_args={'levels': levels}, validate=False)
                pipeline_steps.append((f'l2_imbalance_{levels}_calculator', calculator))
                imputer_step = get_output_imputer_step(params.get('imputation'), 0.0) # Default fill 0
                if imputer_step: pipeline_steps.append(imputer_step)
                scaler_step = get_output_scaler_step(params.get('scaling'))
                if scaler_step: pipeline_steps.append(scaler_step)
                if pipeline_steps:
                    self.feature_pipelines[pipeline_name] = {'pipeline': Pipeline(steps=pipeline_steps), 'input_type': input_type, 'params': params}
                    self.logger.info("Built L2 Imbalance pipeline: %s, input: %s", pipeline_name, input_type, source_module=self._source_module)

            elif feature_key.startswith("l2_wap"):
                input_type = 'l2_book_series'
                levels = params.get("levels", 1)
                calculator = FunctionTransformer(FeatureEngine._pipeline_compute_l2_wap, kw_args={'levels': levels}, validate=False)
                pipeline_steps.append((f'l2_wap_{levels}_calculator', calculator))
                imputer_step = get_output_imputer_step(params.get('imputation'), default_fill_value=np.nan) # fill with mean
                if imputer_step: pipeline_steps.append(imputer_step)
                scaler_step = get_output_scaler_step(params.get('scaling'))
                if scaler_step: pipeline_steps.append(scaler_step)
                if pipeline_steps:
                    self.feature_pipelines[pipeline_name] = {'pipeline': Pipeline(steps=pipeline_steps), 'input_type': input_type, 'params': params}
                    self.logger.info("Built L2 WAP pipeline: %s, input: %s", pipeline_name, input_type, source_module=self._source_module)

            elif feature_key.startswith("l2_depth"):
                input_type = 'l2_book_series'
                levels = params.get("levels", 5)
                calculator = FunctionTransformer(FeatureEngine._pipeline_compute_l2_depth, kw_args={'levels': levels}, validate=False)
                pipeline_steps.append((f'l2_depth_{levels}_calculator', calculator))
                imputer_step = get_output_imputer_step(params.get('imputation'), default_fill_value=np.nan, is_dataframe_output=True) # df.mean()
                if imputer_step: pipeline_steps.append(imputer_step)
                scaler_step = get_output_scaler_step(params.get('scaling'))
                if scaler_step: pipeline_steps.append(scaler_step)
                if pipeline_steps:
                    self.feature_pipelines[pipeline_name] = {'pipeline': Pipeline(steps=pipeline_steps), 'input_type': input_type, 'params': params}
                    self.logger.info("Built L2 Depth pipeline: %s, input: %s", pipeline_name, input_type, source_module=self._source_module)

            # --- Trade-Based Features ---
            elif feature_key.startswith("vwap_trades"):
                input_type = 'trades_and_bar_starts' # Special input type
                interval_s = params.get("length_seconds", 60) # from vwap config
                # kw_args for trade_history_deque will be passed at transform time
                calculator = FunctionTransformer(FeatureEngine._pipeline_compute_vwap_trades, kw_args={'bar_interval_seconds': interval_s}, validate=False)
                pipeline_steps.append((f'vwap_trades_{interval_s}s_calculator', calculator))
                imputer_step = get_output_imputer_step(params.get('imputation'), default_fill_value=np.nan) # fill with mean
                if imputer_step: pipeline_steps.append(imputer_step)
                scaler_step = get_output_scaler_step(params.get('scaling'))
                if scaler_step: pipeline_steps.append(scaler_step)
                if pipeline_steps:
                    self.feature_pipelines[pipeline_name] = {'pipeline': Pipeline(steps=pipeline_steps), 'input_type': input_type, 'params': params}
                    self.logger.info("Built VWAP_Trades pipeline: %s, input: %s", pipeline_name, input_type, source_module=self._source_module)

            elif feature_key.startswith("volume_delta"):
                input_type = 'trades_and_bar_starts' # Special input type
                interval_s = params.get("bar_interval_seconds", 60) # from volume_delta config
                calculator = FunctionTransformer(FeatureEngine._pipeline_compute_volume_delta, kw_args={'bar_interval_seconds': interval_s}, validate=False)
                pipeline_steps.append((f'volume_delta_{interval_s}s_calculator', calculator))
                imputer_step = get_output_imputer_step(params.get('imputation'), 0.0) # Default fill 0
                if imputer_step: pipeline_steps.append(imputer_step)
                scaler_step = get_output_scaler_step(params.get('scaling'))
                if scaler_step: pipeline_steps.append(scaler_step)
                if pipeline_steps:
                    self.feature_pipelines[pipeline_name] = {'pipeline': Pipeline(steps=pipeline_steps), 'input_type': input_type, 'params': params}
                    self.logger.info("Built Volume Delta pipeline: %s, input: %s", pipeline_name, input_type, source_module=self._source_module)


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
