"""Feature engineering for Gal-Friday using a configurable Scikit-learn pipeline approach.

This module provides the `FeatureEngine` class, which is responsible for:
1.  Loading canonical feature definitions from a YAML-based Feature Registry
    (e.g., `config/feature_registry.yaml`).
2.  Processing application-level configuration (from `config.yaml`) to activate
    and override these registry definitions. This allows for customized feature sets
    per bot instance.
3.  Managing historical market data, including OHLCV bars, L2 order book snapshots,
    and raw trade data.
4.  Constructing Scikit-learn pipelines for each activated and configured feature.
    These pipelines handle:
    a.  Input data selection and preparation.
    b.  Core feature calculation, often leveraging `pandas-ta` or custom static methods
        (e.g., `_pipeline_compute_rsi`). These calculator methods are designed for
        robustness, including "Zero NaN" output policies where appropriate (e.g.,
        filling RSI NaNs with 50, or MACD NaNs with 0).
    c.  Optional final output imputation (as a fallback, if a calculator still
        produces NaNs despite its internal handling).
    d.  Centralized output scaling (e.g., StandardScaler, MinMaxScaler) applied
        universally based on feature configuration, ensuring features are ready for
        model consumption.
5.  Executing these pipelines when new market data arrives (typically triggered by
    the close of a new OHLCV bar).
6.  Validating the dictionary of calculated numerical features against a Pydantic model
    (`PublishedFeaturesV1`) to ensure data integrity and contract adherence.
7.  Publishing the validated, numerical features as a dictionary payload within a
    `FEATURES_CALCULATED` event on the PubSub system.

Predictors downstream expect features to be pre-scaled and numerical, as handled by this engine.
"""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd
import pandas_ta as ta
import yaml
from sklearn.base import BaseEstimator, clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from gal_friday.core.events import EventType
from gal_friday.core.feature_models import PublishedFeaturesV1

# Attempt to import FeatureCategory, handle potential circularity if it arises
try:
    from gal_friday.interfaces.feature_engine_interface import FeatureCategory
except ImportError:
    # Define local fallback enum
    class _LocalFeatureCategory(Enum):
        TECHNICAL = "TECHNICAL"
        L2_ORDER_BOOK = "L2_ORDER_BOOK"
        TRADE_DATA = "TRADE_DATA"
        SENTIMENT = "SENTIMENT"
        CUSTOM = "CUSTOM"
        UNKNOWN = "UNKNOWN"

    FeatureCategory = _LocalFeatureCategory  # type: ignore[assignment,misc]


if TYPE_CHECKING:
    from collections.abc import Callable

    from gal_friday.core.pubsub import PubSubManager
    from gal_friday.interfaces.historical_data_service_interface import (
        HistoricalDataService,
    )
    from gal_friday.logger_service import LoggerService

# Type Aliases for common data structures
PandasDataType: TypeAlias = pd.Series | pd.DataFrame | np.ndarray
ScalerType: TypeAlias = StandardScaler | MinMaxScaler | RobustScaler | BaseEstimator

# Constants for Numpy array dimensions
NUMPY_1D_ARRAY_DIM = 1
NUMPY_2D_ARRAY_DIM = 2


@dataclass
class InternalFeatureSpec:
    """Data class for storing internal feature specifications.

    Attributes:
        key: Unique key for the feature. Used for activation via app config and as a
             base for published feature names.
        calculator_type: Defines the core calculation logic (e.g., "rsi", "macd").
                         Maps to a `_pipeline_compute_{calculator_type}` method.
        input_type: Specifies the type of input data required by the calculator
                    (e.g., 'close_series', 'ohlcv_df', 'l2_book_series').
        category: Categorizes the feature (e.g., TECHNICAL, L2_ORDER_BOOK, TRADE_DATA).
        parameters: Dictionary of parameters passed to the feature calculator function.
        imputation: Configuration for the output imputation step in the pipeline
                    (e.g., `{"strategy": "constant", "fill_value": 0.0}`).
                    Applied as a final fallback.
        scaling: Configuration for the output scaling step (e.g., `{"method": "standard"}`).
                 Applied by FeatureEngine.
        description: Human-readable description of the feature and its configuration.
        version: Version string for the feature definition, loaded from the registry.
        output_properties: Dictionary describing expected output characteristics
                           (e.g., `{"value_type": "float", "range": [0, 1]}`).
    """

    key: str
    calculator_type: str
    input_type: str
    category: FeatureCategory = FeatureCategory.TECHNICAL
    parameters: dict[str, Any] = field(default_factory=dict)
    imputation: dict[str, Any] | str | None = None
    scaling: dict[str, Any] | str | None = None
    description: str = ""
    version: str | None = None
    output_properties: dict[str, Any] = field(default_factory=dict)


DEFAULT_FEATURE_REGISTRY_PATH = Path("config/feature_registry.yaml")


class PandasScalerTransformer:
    """A wrapper around sklearn scalers that preserves pandas Series/DataFrame structure.

    This transformer wraps any sklearn scaler (StandardScaler, MinMaxScaler, etc.)
    and ensures that the output maintains the same pandas structure as the input,
    including column names and index.
    """

    def __init__(self, scaler: ScalerType) -> None:
        """Initialize with an sklearn scaler instance.

        Args:
            scaler: An instance of a scikit-learn scaler.
        """
        self.scaler = scaler
        self._feature_names: pd.Index[Any] | str | None = None
        self._index: pd.Index[Any] | None = None

    def fit(
        self,
        x: PandasDataType,
        y: PandasDataType | None = None,
    ) -> PandasScalerTransformer:
        """Fit the scaler and store structure information.

        Args:
            x: The input data to fit the scaler.
            y: Ignored. Present for API consistency.

        Returns:
            The fitted PandasScalerTransformer instance.
        """
        if isinstance(x, pd.DataFrame):
            self._feature_names = x.columns
            self._index = x.index
            self.scaler.fit(x.values)
        elif isinstance(x, pd.Series):
            self._feature_names = x.name
            self._index = x.index
            self.scaler.fit(x.values.reshape(-1, 1))
        else: # Assuming np.ndarray
            self.scaler.fit(x)
        return self

    def transform(self, x: PandasDataType) -> PandasDataType:
        """Transform the data and restore pandas structure.

        Args:
            x: The input data to transform.

        Returns:
            The transformed data, preserving pandas structure if applicable.
        """
        if isinstance(x, pd.DataFrame):
            transformed_values = self.scaler.transform(x.values)
            return pd.DataFrame(
                transformed_values,
                columns=self._feature_names if self._feature_names is not None else x.columns,
                index=x.index,
            )
        if isinstance(x, pd.Series):
            transformed_values = self.scaler.transform(x.values.reshape(-1, 1))
            series_name = self._feature_names if self._feature_names is not None else x.name
            return pd.Series(
                transformed_values.flatten(),
                name=str(series_name) if series_name is not None else None,
                index=x.index,
            )
        return self.scaler.transform(x) # type: ignore[no-any-return]

    def fit_transform(
        self,
        x: PandasDataType,
        y: PandasDataType | None = None,
    ) -> PandasDataType:
        """Fit and transform in one step.

        Args:
            x: The input data to fit and transform.
            y: Ignored. Present for API consistency.

        Returns:
            The transformed data, preserving pandas structure if applicable.
        """
        self.fit(x, y)
        return self.transform(x)


class FeatureEngine:
    """Orchestrates feature calculation based on market data and configurations.

    The FeatureEngine loads canonical feature definitions from a YAML Feature Registry
    (path defined by `DEFAULT_FEATURE_REGISTRY_PATH`). It then uses the application-level
    configuration (passed during initialization, typically from `config.yaml`) to
    determine which features to activate and what customizations (overrides) to apply
    to their registry definitions. This activation and override mechanism is handled
    by the `_extract_feature_configs` method.

    For each activated feature (now represented as an `InternalFeatureSpec`), the
    `_build_feature_pipelines` method constructs a Scikit-learn `Pipeline`.
    This pipeline typically includes:
    1.  Input data selection and preparation (e.g., input imputation for close price series).
    2.  The core feature calculation logic, delegated to a static `_pipeline_compute_...`
        method corresponding to the feature's `calculator_type`. These methods are
        designed for robust NaN handling, aiming to provide sensible default values
        if a feature cannot be computed (e.g., RSI defaulting to 50.0).
    3.  An optional output imputation step, configured via the `imputation` field in
        the `InternalFeatureSpec`. This serves as a final fallback if a calculator,
        despite its internal NaN handling, still yields NaNs.
    4.  A centralized output scaling step, configured via the `scaling` field in the
        `InternalFeatureSpec`. This ensures that features are scaled (e.g., using
        StandardScaler or MinMaxScaler) before being published, making them ready for
        direct consumption by machine learning models.

    When new market data triggers `_calculate_and_publish_features`, these pipelines are
    executed. The resulting numerical feature values are collected into a dictionary.
    This dictionary is then validated against the `PublishedFeaturesV1` Pydantic model
    to ensure data integrity and adherence to the defined contract. If valid, the
    features are published as `pydantic_model.model_dump()` within a `FEATURES_CALCULATED`
    event on the PubSub system. Downstream services, like predictors, thus receive
    pre-scaled, validated, numerical features.
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
            config: Overall application configuration dictionary. `FeatureEngine` uses
                `config['feature_engine']['features']` for feature activation and
                overrides, and `config['feature_engine']` for other settings.
                Feature definitions are primarily loaded from the YAML Feature Registry
                invoked by `_extract_feature_configs`.
            pubsub_manager: Instance of PubSubManager for event handling.
            logger_service: Logging service instance.
            historical_data_service: Optional service for fetching historical data.
        """
        self.config = config
        self.pubsub_manager = pubsub_manager
        self.logger = logger_service
        self.historical_data_service = historical_data_service
        self._source_module = self.__class__.__name__

        self._feature_configs: dict[str, InternalFeatureSpec] = {}
        self._extract_feature_configs()

        self._feature_handlers: dict[
            str,
            Callable[[dict[str, Any], dict[str, Any]], dict[str, str] | None],
        ] = {}


        self.ohlcv_history: dict[str, pd.DataFrame] = defaultdict(
            lambda: pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
            ).astype(
                {
                    "open": "object",
                    "high": "object",
                    "low": "object",
                    "close": "object",
                    "volume": "object",
                },
            ),
        )

        self.l2_books: dict[str, dict[str, Any]] = defaultdict(dict)

        trade_history_maxlen = self.config.get("feature_engine", {}).get(
            "trade_history_maxlen",
            2000,
        )
        self.trade_history: dict[str, deque[Any]] = defaultdict(
            lambda: deque(maxlen=trade_history_maxlen),
        )

        self.feature_pipelines: dict[str, dict[str, Any]] = {}
        self._build_feature_pipelines()

        self.logger.info("FeatureEngine initialized.", source_module=self._source_module)

    def _determine_calculator_type_and_input( # noqa: PLR0911, PLR0912
        self,
        feature_key: str,
        raw_cfg: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        """Determine calculator type and input type from feature key and raw config.

        This method attempts to infer the type of calculation (e.g., "rsi", "macd")
        and the type of input data required (e.g., "close_series", "ohlcv_df")
        primarily by inspecting the `feature_key` string. It also allows for explicit
        overrides if `calculator_type` (or `type`) and `input_type` are specified
        directly in the raw feature configuration dictionary.

        Args:
            feature_key: The original key for the feature (e.g., "rsi_14").
            raw_cfg: The raw dictionary configuration for this specific feature.

        Returns:
            A tuple `(calculator_type, input_type)`, where both can be `None` if
            determination fails.
        """
        calc_type = raw_cfg.get("calculator_type", raw_cfg.get("type"))
        input_type = raw_cfg.get("input_type")

        if calc_type and input_type:
            return str(calc_type), str(input_type)

        key_lower = feature_key.lower()
        if "rsi" in key_lower:
            return "rsi", "close_series"
        if "macd" in key_lower:
            return "macd", "close_series"
        if "bbands" in key_lower:
            return "bbands", "close_series"
        if "roc" in key_lower:
            return "roc", "close_series"
        if "atr" in key_lower:
            return "atr", "ohlcv_df"
        if "stdev" in key_lower:
            return "stdev", "close_series"
        if "vwap_ohlcv" in key_lower:
            return "vwap_ohlcv", "ohlcv_df"
        if "l2_spread" in key_lower:
            return "l2_spread", "l2_book_series"
        if "l2_imbalance" in key_lower:
            return "l2_imbalance", "l2_book_series"
        if "l2_wap" in key_lower:
            return "l2_wap", "l2_book_series"
        if "l2_depth" in key_lower:
            return "l2_depth", "l2_book_series"
        if "vwap_trades" in key_lower:
            return "vwap_trades", "trades_and_bar_starts"
        if "volume_delta" in key_lower:
            return "volume_delta", "trades_and_bar_starts"

        self.logger.warning(
            "Could not determine calculator_type or input_type for feature key: %s",
            feature_key,
            source_module=self._source_module,
        )
        return None, None

    def _extract_feature_configs(self) -> None:  # noqa: PLR0912
        """Initialize `self._feature_configs` by loading feature definitions."""
        registry_definitions = self._load_feature_registry(DEFAULT_FEATURE_REGISTRY_PATH)
        app_feature_config = self.config.get("feature_engine", {}).get("features", {})

        final_parsed_specs: dict[str, InternalFeatureSpec] = {}

        if isinstance(app_feature_config, list):
            for key in app_feature_config:
                if not isinstance(key, str):
                    self.logger.warning(
                        "Feature activation list contains non-string item: %s. Skipping.",
                        key,
                        source_module=self._source_module,
                    )
                    continue
                if key not in registry_definitions:
                    self.logger.warning(
                        "Feature key '%s' from app config not found in registry. Skipping.",
                        key,
                        source_module=self._source_module,
                    )
                    continue

                feature_def_from_registry = registry_definitions[key]
                if not isinstance(feature_def_from_registry, dict):
                    self.logger.warning(
                        "Registry definition for '%s' is not a dictionary. Skipping.",
                        key,
                        source_module=self._source_module,
                    )
                    continue

                spec_result = self._parse_single_feature_definition(
                    key,
                    feature_def_from_registry.copy(),
                )
                if spec_result is not None:
                    final_parsed_specs[key] = spec_result

        elif isinstance(app_feature_config, dict):
            for key, overrides_or_activation in app_feature_config.items():
                if not isinstance(overrides_or_activation, dict):
                    self.logger.warning(
                        "Override/activation config for feature '%s' is not a dict. Skipping.",
                        key,
                        source_module=self._source_module,
                    )
                    continue

                base_config = registry_definitions.get(key)
                final_config_dict: dict[str, Any]

                if base_config:
                    if not isinstance(base_config, dict):
                        self.logger.warning(
                            "Registry definition for '%s' is not a dictionary. "
                            "Skipping override.",
                            key,
                            source_module=self._source_module,
                        )
                        continue
                    final_config_dict = self._deep_merge_configs(
                        base_config.copy(),
                        overrides_or_activation,
                    )
                else:
                    self.logger.info(
                        "Feature '%s' not found in registry, "
                        "treating as ad-hoc definition from app config.",
                        key,
                        source_module=self._source_module,
                    )
                    final_config_dict = overrides_or_activation.copy()
                    if "calculator_type" not in final_config_dict or \
                       "input_type" not in final_config_dict:
                        self.logger.warning(
                            "Ad-hoc feature '%s' missing 'calculator_type' or "
                            "'input_type'. Skipping.",
                            key,
                            source_module=self._source_module,
                        )
                        continue

                spec_result = self._parse_single_feature_definition(key, final_config_dict)
                if spec_result is not None:
                    final_parsed_specs[key] = spec_result
        else:
            self.logger.warning(
                "App-level 'features' config is neither a list nor a dict. "
                "No features will be configured based on it.",
                source_module=self._source_module,
            )

        self._feature_configs = final_parsed_specs
        if not self._feature_configs:
            self.logger.warning(
                "No features were successfully parsed or activated. "
                "FeatureEngine might not produce any features.",
                source_module=self._source_module,
            )

    def _deep_merge_configs(
        self,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        """Deeply merge override dict into base dict."""
        merged = base.copy()
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _parse_single_feature_definition(
        self,
        feature_key: str,
        config_dict: dict[str, Any],
    ) -> InternalFeatureSpec | None:
        """Parse a single, consolidated feature configuration dictionary."""
        calculator_type_any = config_dict.get("calculator_type", config_dict.get("type"))
        input_type_any = config_dict.get("input_type")
        calculator_type: str | None = None
        input_type: str | None = None

        if calculator_type_any is not None:
            calculator_type = str(calculator_type_any)
        if input_type_any is not None:
            input_type = str(input_type_any)


        if not calculator_type or not input_type:
            inferred_calc_type, inferred_input_type = \
                self._determine_calculator_type_and_input(feature_key, config_dict)
            if not calculator_type:
                calculator_type = inferred_calc_type
            if not input_type:
                input_type = inferred_input_type

            if not calculator_type or not input_type:
                self.logger.warning(
                    "Could not determine calculator_type or input_type for feature '%s' "
                    "even after inference. Skipping.",
                    feature_key,
                    source_module=self._source_module,
                )
                return None
        final_calculator_type = str(calculator_type)
        final_input_type = str(input_type)


        parameters = config_dict.get("parameters", config_dict.get("params", {}))
        if not isinstance(parameters, dict):
            parameters = {}

        common_param_keys = [
            "period", "length", "fast", "slow", "signal", "levels",
            "std_dev", "length_seconds", "bar_interval_seconds",
        ]
        for common_param_key in common_param_keys:
            if common_param_key in config_dict and common_param_key not in parameters:
                parameters[common_param_key] = config_dict[common_param_key]

        imputation_cfg = config_dict.get("imputation")
        scaling_cfg = config_dict.get("scaling")
        description = config_dict.get(
            "description",
            f"{final_calculator_type} feature based on {feature_key}",
        )
        version = config_dict.get("version")
        output_properties = config_dict.get("output_properties", {})

        category_str = str(config_dict.get("category", "TECHNICAL")).upper()
        try:
            category = FeatureCategory[category_str]
        except KeyError:
            self.logger.warning(
                "Invalid FeatureCategory '%s' for feature '%s'. Defaulting to TECHNICAL.",
                category_str,
                feature_key,
                source_module=self._source_module,
            )
            category = FeatureCategory.TECHNICAL

        spec = InternalFeatureSpec(
            key=feature_key,
            calculator_type=final_calculator_type,
            input_type=final_input_type,
            category=category,
            parameters=parameters,
            imputation=imputation_cfg,
            scaling=scaling_cfg,
            description=description,
            version=str(version) if version is not None else None,
            output_properties=output_properties if isinstance(output_properties, dict) else {},
        )
        self.logger.info(
            "Successfully parsed feature spec for key: '%s' (Calc: %s, Input: %s)",
            feature_key,
            final_calculator_type,
            final_input_type,
            source_module=self._source_module,
        )
        return spec

    def _load_feature_registry(self, registry_path: Path) -> dict[str, Any]:
        """Load feature definitions from the specified YAML feature registry file."""
        if not registry_path.exists():
            self.logger.error(
                "Feature registry file not found: %s",
                registry_path,
                source_module=self._source_module,
            )
            return {}

        try:
            with registry_path.open("r") as f:
                registry_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.logger.exception(
                "Error parsing YAML in feature registry %s: %s",
                registry_path,
                e,
                source_module=self._source_module,
            )
            return {}
        except OSError as e:
            self.logger.exception(
                "OS error loading feature registry %s: %s",
                registry_path,
                e,
                source_module=self._source_module,
            )
            return {}


        if not isinstance(registry_data, dict):
            self.logger.error(
                "Feature registry %s content is not a dictionary.",
                registry_path,
                source_module=self._source_module,
            )
            return {}

        self.logger.info(
            "Successfully loaded %s feature definitions from %s.",
            len(registry_data),
            registry_path,
            source_module=self._source_module,
        )
        return registry_data

    def _build_feature_pipelines(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Construct Scikit-learn pipelines for each feature."""
        self.logger.info("Building feature pipelines...", source_module=self._source_module)

        def _fillna_constant_val(x: PandasDataType, fill_val: Any) -> PandasDataType:  # noqa: ANN401
            return x.fillna(fill_val)

        def _fillna_with_mean(x: PandasDataType) -> PandasDataType:
            return x.fillna(x.mean()) if not x.empty else x # type: ignore[operator]

        def _fillna_with_median(x: PandasDataType) -> PandasDataType:
            return x.fillna(x.median()) if not x.empty else x # type: ignore[operator]


        def get_output_imputer_step(
            imputation_cfg: dict[str, Any] | str | None,
            default_fill_value: float = 0.0,
            spec_key: str = "",
        ) -> tuple[str, FunctionTransformer] | None:
            """Create a Scikit-learn compatible imputation step."""
            if imputation_cfg == "passthrough":
                self.logger.debug(
                    "Imputation set to 'passthrough' for %s.",
                    spec_key,
                    source_module=self._source_module,
                )
                return None

            strategy = "default"
            current_fill_value = default_fill_value

            if isinstance(imputation_cfg, dict):
                strategy = imputation_cfg.get("strategy", "constant")
                if strategy == "constant":
                    current_fill_value = imputation_cfg.get("fill_value", default_fill_value)
            elif imputation_cfg is None:
                pass
            else:
                self.logger.warning(
                    "Unrecognized imputation config for %s: %s. No imputer added.",
                    spec_key,
                    imputation_cfg,
                    source_module=self._source_module,
                )
                return None

            step_name_suffix = ""
            transform_func: Callable[[PandasDataType], PandasDataType]

            if strategy == "constant":
                step_name_suffix = f"const_{current_fill_value}"
                def func_fillna_constant(x: PandasDataType) -> PandasDataType:
                    return _fillna_constant_val(x, current_fill_value)
                transform_func = func_fillna_constant
            elif strategy == "mean":
                step_name_suffix = "mean"
                transform_func = _fillna_with_mean
            elif strategy == "median":
                step_name_suffix = "median"
                transform_func = _fillna_with_median
            elif strategy == "default":
                step_name_suffix = f"default_fill_{current_fill_value}"
                def func_fillna_default(x: PandasDataType) -> PandasDataType:
                    return _fillna_constant_val(x, current_fill_value)
                transform_func = func_fillna_default
            else:
                self.logger.warning(
                    "Unknown imputation strategy '%s' for %s. No imputer added.",
                    strategy,
                    spec_key,
                    source_module=self._source_module,
                )
                return None

            log_fill_val_display = current_fill_value if strategy in {"constant", "default"} \
                else "N/A"
            log_strategy_display = strategy if strategy != "default" else \
                                   f"default_fill({current_fill_value})"

            self.logger.debug(
                "Using output imputer strategy '%s' (fill: %s) for %s",
                log_strategy_display,
                log_fill_val_display,
                spec_key,
                source_module=self._source_module,
            )
            return (
                f"{spec_key}_output_imputer_{step_name_suffix}",
                FunctionTransformer(transform_func, validate=False),
            )

        def get_output_scaler_step(
            scaling_cfg: dict[str, Any] | str | None,
            spec_key: str = "",
        ) -> tuple[str, PandasScalerTransformer] | None:
            """Create a Scikit-learn compatible scaling step."""
            if scaling_cfg == "passthrough" or scaling_cfg is None:
                log_msg = "Scaling set to 'passthrough' for %s." if scaling_cfg == "passthrough" \
                    else "No scaling configured or default 'None' for %s."
                self.logger.debug(log_msg, spec_key, source_module=self._source_module)
                return None

            scaler_instance: ScalerType = StandardScaler()
            scaler_name_suffix = "StandardScaler"

            if isinstance(scaling_cfg, dict):
                method = scaling_cfg.get("method", "standard")
                if method == "minmax":
                    scaler_instance = MinMaxScaler(
                        feature_range=scaling_cfg.get("feature_range", (0, 1)),
                    )
                    min_val, max_val = scaler_instance.feature_range # type: ignore
                    scaler_name_suffix = f"MinMaxScaler_({min_val},{max_val})"
                elif method == "robust":
                    scaler_instance = RobustScaler(
                        quantile_range=scaling_cfg.get("quantile_range", (25.0, 75.0)),
                    )
                    q_low, q_high = scaler_instance.quantile_range # type: ignore
                    scaler_name_suffix = f"RobustScaler_({q_low},{q_high})"
                elif method != "standard":
                    self.logger.warning(
                        "Unknown scaling method '%s' for %s. Using StandardScaler.",
                        method,
                        spec_key,
                        source_module=self._source_module,
                    )
            elif isinstance(scaling_cfg, str) and scaling_cfg not in [
                "standard", "passthrough",
            ]:
                self.logger.warning(
                    "Simple string for scaling method '%s' for %s is ambiguous. "
                    "Use dict config or 'passthrough'. Defaulting to StandardScaler.",
                    scaling_cfg,
                    spec_key,
                    source_module=self._source_module,
                )

            self.logger.debug(
                "Using %s for scaling for %s",
                type(scaler_instance).__name__,
                spec_key,
                source_module=self._source_module,
            )
            return (
                f"{spec_key}_output_scaler_{scaler_name_suffix}",
                PandasScalerTransformer(scaler_instance),
            )

        for spec in self._feature_configs.values():
            pipeline_steps = []
            pipeline_name = f"{spec.key}_pipeline"

            if spec.input_type == "close_series":
                self.logger.debug(
                    "Adding standard input imputer (mean) for %s",
                    spec.key,
                    source_module=self._source_module,
                )
                pipeline_steps.append(
                    (f"{spec.key}_input_imputer", SimpleImputer(strategy="mean")),
                )

            calculator_func = getattr(
                FeatureEngine,
                f"_pipeline_compute_{spec.calculator_type}",
                None,
            )
            if not calculator_func:
                self.logger.error(
                    "No _pipeline_compute function for calc_type: %s (feature: %s)",
                    spec.calculator_type,
                    spec.key,
                    source_module=self._source_module,
                )
                continue

            calc_kw_args: dict[str, Any] = {}
            if spec.calculator_type in ["rsi", "roc", "stdev"]:
                default_period = 14 if spec.calculator_type == "rsi" \
                    else 10 if spec.calculator_type == "roc" else 20
                calc_kw_args["period"] = int(spec.parameters.get("period", default_period))
                if spec.parameters.get("period") is None:
                    self.logger.debug(
                        "Using default period %s for %s ('%s')",
                        calc_kw_args["period"],
                        spec.calculator_type,
                        spec.key,
                        source_module=self._source_module,
                    )
            elif spec.calculator_type == "macd":
                calc_kw_args["fast"] = int(spec.parameters.get("fast", 12))
                calc_kw_args["slow"] = int(spec.parameters.get("slow", 26))
                calc_kw_args["signal"] = int(spec.parameters.get("signal", 9))
                if any(p not in spec.parameters for p in ["fast", "slow", "signal"]):
                    self.logger.debug(
                        "Using default MACD params (f:%s,s:%s,sig:%s) for %s",
                        calc_kw_args["fast"],
                        calc_kw_args["slow"],
                        calc_kw_args["signal"],
                        spec.key,
                        source_module=self._source_module,
                    )
            elif spec.calculator_type == "bbands":
                calc_kw_args["length"] = int(spec.parameters.get("length", 20))
                calc_kw_args["std_dev"] = float(spec.parameters.get("std_dev", 2.0))
                if "length" not in spec.parameters or "std_dev" not in spec.parameters:
                    self.logger.debug(
                        "Using default BBands params (l:%s,s:%.1f) for %s",
                        calc_kw_args["length"],
                        calc_kw_args["std_dev"],
                        spec.key,
                        source_module=self._source_module,
                    )
            elif spec.calculator_type == "atr":
                calc_kw_args["length"] = int(spec.parameters.get("length", 14))
                if "length" not in spec.parameters:
                    self.logger.debug(
                        "Using default ATR length %s for %s",
                        calc_kw_args["length"],
                        spec.key,
                        source_module=self._source_module,
                    )
            elif spec.calculator_type == "vwap_ohlcv":
                calc_kw_args["length"] = int(spec.parameters.get("length", 14))
                if "length" not in spec.parameters:
                    self.logger.debug(
                        "Using default VWAP_OHLCV length %s for %s",
                        calc_kw_args["length"],
                        spec.key,
                        source_module=self._source_module,
                    )
            elif spec.calculator_type in ["l2_imbalance", "l2_depth", "l2_wap"]:
                default_levels = 5
                if spec.calculator_type == "l2_wap":
                    default_levels = 1
                elif spec.calculator_type == "l2_spread":
                    default_levels = 0

                if default_levels > 0:
                    calc_kw_args["levels"] = int(spec.parameters.get("levels", default_levels))
                    if "levels" not in spec.parameters:
                        self.logger.debug(
                            "Using default levels %s for %s ('%s')",
                            calc_kw_args["levels"],
                            spec.calculator_type,
                            spec.key,
                            source_module=self._source_module,
                        )
            elif spec.calculator_type in ["vwap_trades", "volume_delta"]:
                default_interval = 60
                calc_kw_args["bar_interval_seconds"] = int(spec.parameters.get(
                    "bar_interval_seconds",
                    spec.parameters.get("length_seconds", default_interval),
                ))
                if "bar_interval_seconds" not in spec.parameters and \
                   "length_seconds" not in spec.parameters:
                    self.logger.debug(
                        "Using default bar_interval_seconds %s for %s ('%s')",
                        calc_kw_args["bar_interval_seconds"],
                        spec.calculator_type,
                        spec.key,
                        source_module=self._source_module,
                    )

            pipeline_steps.append(
                (
                    f"{spec.key}_calculator",
                    FunctionTransformer(calculator_func, kw_args=calc_kw_args, validate=False),
                ),
            )

            default_fill = 0.0
            if spec.calculator_type == "rsi":
                default_fill = 50.0
            elif spec.calculator_type in [
                "atr", "vwap_ohlcv", "l2_wap", "vwap_trades", "stdev",
            ]:
                default_fill = np.nan

            imputer_step = get_output_imputer_step(
                spec.imputation,
                default_fill_value=default_fill,
                spec_key=spec.key,
            )
            if imputer_step:
                pipeline_steps.append(imputer_step)

            scaler_step = get_output_scaler_step(spec.scaling, spec_key=spec.key)
            if scaler_step:
                pipeline_steps.append(scaler_step)

            if pipeline_steps:
                final_pipeline = Pipeline(steps=pipeline_steps)
                final_pipeline.set_output(transform="pandas")
                self.feature_pipelines[pipeline_name] = {
                    "pipeline": final_pipeline,
                    "input_type": spec.input_type,
                    "params": spec.parameters,
                    "spec": spec,
                }
                self.logger.info(
                    "Built pipeline: %s with steps: %s, input: %s",
                    pipeline_name,
                    [s[0] for s in pipeline_steps],
                    spec.input_type,
                    source_module=self._source_module,
                )

    def _handle_ohlcv_update(
        self,
        trading_pair: str,
        ohlcv_payload: dict[str, Any],
    ) -> None:
        """Parse and store an OHLCV update."""
        try:
            timestamp_str = ohlcv_payload.get("timestamp_bar_start")
            if not timestamp_str:
                self.logger.warning(
                    "Missing 'timestamp_bar_start' in OHLCV payload for %s",
                    trading_pair,
                    source_module=self._source_module,
                    context={"payload": ohlcv_payload},
                )
                return

            bar_timestamp = pd.to_datetime(str(timestamp_str), utc=True)

            new_bar_data = {
                "open": Decimal(str(ohlcv_payload["open"])),
                "high": Decimal(str(ohlcv_payload["high"])),
                "low": Decimal(str(ohlcv_payload["low"])),
                "close": Decimal(str(ohlcv_payload["close"])),
                "volume": Decimal(str(ohlcv_payload["volume"])),
            }

            df = self.ohlcv_history[trading_pair]
            new_row_df = pd.DataFrame([new_bar_data], index=[bar_timestamp])
            new_row_df.index.name = "timestamp_bar_start"

            for col in df.columns:
                if col not in new_row_df:
                    new_row_df[col] = pd.NA
            new_row_df = new_row_df[df.columns]

            if bar_timestamp not in df.index:
                df = pd.concat([df, new_row_df])
            else:
                df.loc[bar_timestamp] = new_row_df.iloc[0]
                self.logger.debug(
                    "Updated existing OHLCV bar for %s at %s",
                    trading_pair,
                    bar_timestamp,
                    source_module=self._source_module,
                )

            df.sort_index(inplace=True)

            min_hist = self._get_min_history_required()
            required_length = min_hist + 50
            if len(df) > required_length:
                df = df.iloc[-required_length:]

            self.ohlcv_history[trading_pair] = df
            self.logger.debug(
                "Processed OHLCV update for %s. History size: %s",
                trading_pair,
                len(df),
                source_module=self._source_module,
            )

        except KeyError as e:
            self.logger.exception(
                "Missing key '%s' in OHLCV payload for %s.",
                str(e),
                trading_pair,
                source_module=self._source_module,
                context={"payload": ohlcv_payload},
            )
        except (ValueError, TypeError) as e:
            self.logger.exception(
                "Data conversion error in OHLCV payload for %s: %s",
                trading_pair,
                str(e),
                source_module=self._source_module,
                context={"payload": ohlcv_payload},
            )
        except Exception as e:
            self.logger.exception(
                "Unexpected error handling OHLCV update for %s: %s",
                trading_pair,
                str(e),
                source_module=self._source_module,
                context={"payload": ohlcv_payload},
            )

    def _handle_l2_update(self, trading_pair: str, l2_payload: dict[str, Any]) -> None:
        """Parse and store an L2 order book update."""
        try:
            raw_bids = l2_payload.get("bids")
            raw_asks = l2_payload.get("asks")

            if not isinstance(raw_bids, list) or not isinstance(raw_asks, list):
                self.logger.warning(
                    "L2 bids/asks are not lists for %s. Payload: %s",
                    trading_pair,
                    l2_payload,
                    source_module=self._source_module,
                )
                return

            processed_bids = []
            for i, bid_level in enumerate(raw_bids):
                if (
                    isinstance(bid_level, list | tuple)
                    and len(bid_level) == self._EXPECTED_L2_LEVEL_LENGTH
                ):
                    try:
                        processed_bids.append(
                            [Decimal(str(bid_level[0])), Decimal(str(bid_level[1]))],
                        )
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            "Error converting L2 bid level %s for %s: %s - %s",
                            i,
                            trading_pair,
                            bid_level,
                            e,
                            source_module=self._source_module,
                        )
                        continue
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
                        processed_asks.append(
                            [Decimal(str(ask_level[0])), Decimal(str(ask_level[1]))],
                        )
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            "Error converting L2 ask level %s for %s: %s - %s",
                            i,
                            trading_pair,
                            ask_level,
                            e,
                            source_module=self._source_module,
                        )
                        continue
                else:
                    self.logger.warning(
                        "Malformed L2 ask level %s for %s: %s",
                        i,
                        trading_pair,
                        ask_level,
                        source_module=self._source_module,
                    )

            self.l2_books[trading_pair] = {
                "bids": processed_bids,
                "asks": processed_asks,
                "timestamp": pd.to_datetime(
                    str(l2_payload.get("timestamp_exchange")) \
                        if l2_payload.get("timestamp_exchange") else datetime.utcnow(),
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

        except KeyError as e:
            self.logger.exception(
                "Missing key '%s' in L2 payload for %s.",
                str(e),
                trading_pair,
                source_module=self._source_module,
                context={"payload": l2_payload},
            )
        except Exception as e:
            self.logger.exception(
                "Unexpected error handling L2 update for %s: %s",
                trading_pair,
                str(e),
                source_module=self._source_module,
                context={"payload": l2_payload},
            )

    async def _handle_trade_event(self, event_dict: dict[str, Any]) -> None:
        """Handle incoming raw trade events and store them."""
        payload = event_dict.get("payload")
        if not payload:
            self.logger.warning(
                "Trade event missing payload.",
                context=event_dict,
                source_module=self._source_module,
            )
            return

        trading_pair_any = payload.get("trading_pair")
        if not trading_pair_any:
            self.logger.warning(
                "Trade event payload missing trading_pair.",
                context=payload,
                source_module=self._source_module,
            )
            return
        trading_pair = str(trading_pair_any)


        try:
            trade_timestamp_str = payload.get("timestamp_exchange")
            price_str = payload.get("price")
            volume_str = payload.get("volume")
            side_any = payload.get("side")

            if not all([trade_timestamp_str, price_str, volume_str, side_any]):
                self.logger.warning(
                    "Trade event for %s is missing required fields "
                    "(timestamp, price, volume, or side).",
                    trading_pair,
                    context=payload,
                    source_module=self._source_module,
                )
                return

            trade_data = {
                "timestamp": pd.to_datetime(str(trade_timestamp_str), utc=True),
                "price": Decimal(str(price_str)),
                "volume": Decimal(str(volume_str)),
                "side": str(side_any).lower(),
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

        except KeyError as e:
            self.logger.exception(
                "Missing key '%s' in trade event payload for %s.",
                str(e),
                trading_pair,
                source_module=self._source_module,
                context=payload,
            )
        except (ValueError, TypeError) as e:
            self.logger.exception(
                "Data conversion error in trade event payload for %s: %s",
                trading_pair,
                str(e),
                source_module=self._source_module,
                context=payload,
            )
        except Exception as e:
            self.logger.exception(
                "Unexpected error handling trade event for %s: %s",
                trading_pair,
                str(e),
                source_module=self._source_module,
                context=payload,
            )

    def _get_min_history_required(self) -> int:
        """Determine the minimum required history size for TA calculations."""
        min_size = 1
        periods: list[int] = []
        for spec in self._feature_configs.values():
            period_val: Any = None
            if spec.calculator_type in ["rsi", "roc"]:
                period_val = spec.parameters.get(
                    "period",
                    14 if spec.calculator_type == "rsi" else 10,
                )
            elif spec.calculator_type in ["bbands", "vwap_ohlcv", "atr", "stdev"]:
                period_val = spec.parameters.get(
                    "length",
                    20 if spec.calculator_type == "bbands" else 14,
                )
            if period_val is not None:
                 try:
                    periods.append(int(period_val))
                 except (ValueError, TypeError):
                    self.logger.warning(
                        "Invalid period/length value '%s' for feature %s. Using default.",
                        period_val,
                        spec.key,
                        source_module=self._source_module,
                    )
                    periods.append(14)


        if periods:
            min_size = max(periods) * 3

        return max(100, min_size)

    def _get_period_from_config( # pylint: disable=unused-private-member
        self,
        feature_name: str,
        field_name: str,
        default_value: int,
    ) -> int:
        """Retrieve the period from config for a specific feature."""
        feature_spec = self._feature_configs.get(feature_name)
        if feature_spec and isinstance(feature_spec, InternalFeatureSpec):
            period_value = feature_spec.parameters.get(field_name, default_value)
            if isinstance(period_value, int) and period_value > 0:
                return period_value
        self.logger.warning(
            "Configuration for feature '%s' not found or invalid when trying to get '%s'. "
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
                self._handle_trade_event,
            )
            self.logger.info(
                "FeatureEngine started and subscribed to MARKET_DATA_OHLCV, "
                "MARKET_DATA_L2, and MARKET_DATA_TRADE events.",
                source_module=self._source_module,
            )
        except Exception as e:
            self.logger.exception(
                "Error during FeatureEngine start and subscription: %s",
                str(e),
                source_module=self._source_module,
            )

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
                self._handle_trade_event,
            )
            self.logger.info(
                "FeatureEngine stopped and unsubscribed from market data events.",
                source_module=self._source_module,
            )
        except Exception as e:
            self.logger.exception(
                "Error during FeatureEngine stop and unsubscription: %s",
                str(e),
                source_module=self._source_module,
            )

    async def process_market_data(self, market_data_event_dict: dict[str, Any]) -> None:
        """Process market data to generate features.

        Args:
            market_data_event_dict: Market data event dictionary.
        """
        event_type_any = market_data_event_dict.get("event_type")
        payload = market_data_event_dict.get("payload")
        source_module_event = market_data_event_dict.get("source_module")

        if not event_type_any or not payload:
            self.logger.warning(
                "Received market data event with missing event_type or payload.",
                source_module=self._source_module,
                context={"original_event": market_data_event_dict},
            )
            return

        event_type_str = str(event_type_any)

        trading_pair_any = payload.get("trading_pair")
        if not trading_pair_any:
            self.logger.warning(
                "Market data event (type: %s) missing trading_pair.",
                event_type_str,
                source_module=self._source_module,
                context={"original_event": market_data_event_dict},
            )
            return
        trading_pair = str(trading_pair_any)


        self.logger.debug(
            "Processing event %s for %s from %s",
            event_type_str,
            trading_pair,
            source_module_event,
            source_module=self._source_module,
        )

        if event_type_str == EventType.MARKET_DATA_OHLCV.name:
            self._handle_ohlcv_update(trading_pair, payload)
            timestamp_bar_start = payload.get("timestamp_bar_start")
            if timestamp_bar_start:
                await self._calculate_and_publish_features(
                    trading_pair,
                    str(timestamp_bar_start),
                )
            else:
                self.logger.warning(
                    "OHLCV event for %s missing 'timestamp_bar_start', "
                    "cannot calculate features.",
                    trading_pair,
                    source_module=self._source_module,
                    context={"payload": payload},
                )
        elif event_type_str == EventType.MARKET_DATA_L2.name:
            self._handle_l2_update(trading_pair, payload)
        elif event_type_str == EventType.MARKET_DATA_TRADE.name:
            await self._handle_trade_event(market_data_event_dict)
        else:
            self.logger.warning(
                "Received unknown market data event type: %s for %s",
                event_type_str,
                trading_pair,
                source_module=self._source_module,
                context={"original_event": market_data_event_dict},
            )

    @staticmethod
    def _pipeline_compute_rsi(data: pd.Series, period: int) -> pd.Series:
        """Compute RSI using pandas-ta.

        Args:
            data: Input Series (typically close prices).
            period: The period for RSI calculation.

        Returns:
            A Series containing the RSI values.
        """
        if not isinstance(data, pd.Series):
            return pd.Series(dtype="float64", name=f"rsi_{period}")
        rsi_series = data.astype("float64").ta.rsi(length=period)
        if rsi_series is None:
             return pd.Series(dtype="float64", name=f"rsi_{period}").fillna(50.0)
        rsi_series = rsi_series.fillna(50.0)
        rsi_series.name = f"rsi_{period}"
        return rsi_series.astype("float64")

    @staticmethod
    def _pipeline_compute_macd(
        data: pd.Series,
        fast: int,
        slow: int,
        signal: int,
    ) -> pd.DataFrame:
        """Compute MACD using pandas-ta.

        Args:
            data: Input Series (typically close prices).
            fast: Fast period for MACD.
            slow: Slow period for MACD.
            signal: Signal period for MACD.

        Returns:
            A DataFrame with MACD, histogram, and signal lines.
        """
        if not isinstance(data, pd.Series):
            return pd.DataFrame(dtype="float64")
        macd_df = data.astype("float64").ta.macd(fast=fast, slow=slow, signal=signal)
        if macd_df is not None:
            macd_df = macd_df.fillna(0.0)
            return macd_df.astype("float64")
        return pd.DataFrame(dtype="float64")


    @staticmethod
    def _fillna_bbands(bbands_df: pd.DataFrame, close_prices: pd.Series) -> pd.DataFrame:
        """Helper to fill NaNs in Bollinger Bands results.

        Middle band NaN is filled with close price. Lower/Upper NaNs also with close price.

        Args:
            bbands_df: DataFrame with Bollinger Bands columns.
            close_prices: Series of close prices for filling.

        Returns:
            DataFrame with NaNs filled.
        """
        if bbands_df is None:
            return pd.DataFrame(dtype="float64")

        aligned_close_prices = close_prices.reindex(bbands_df.index)

        middle_col = next((col for col in bbands_df.columns if col.startswith("BBM_")), None)
        lower_col = next((col for col in bbands_df.columns if col.startswith("BBL_")), None)
        upper_col = next((col for col in bbands_df.columns if col.startswith("BBU_")), None)

        if middle_col:
            bbands_df[middle_col] = bbands_df[middle_col].fillna(aligned_close_prices)
        if lower_col:
            bbands_df[lower_col] = bbands_df[lower_col].fillna(aligned_close_prices)
        if upper_col:
            bbands_df[upper_col] = bbands_df[upper_col].fillna(aligned_close_prices)

        for col in bbands_df.columns:
            if col not in [middle_col, lower_col, upper_col] and bbands_df[col].isna().any():
                bbands_df[col] = bbands_df[col].fillna(0.0)
        return bbands_df


    @staticmethod
    def _pipeline_compute_bbands(
        data: pd.Series,
        length: int,
        std_dev: float,
    ) -> pd.DataFrame:
        """Compute Bollinger Bands using pandas-ta.

        Args:
            data: Input Series (typically close prices).
            length: The period for BBands calculation.
            std_dev: The number of standard deviations.

        Returns:
            A DataFrame with lower, middle, upper bands.
        """
        if not isinstance(data, pd.Series):
            return pd.DataFrame(dtype="float64")
        bbands_df = data.astype("float64").ta.bbands(length=length, std=std_dev)
        if bbands_df is not None:
            bbands_df = FeatureEngine._fillna_bbands(bbands_df, data.astype("float64"))
            return bbands_df.astype("float64")
        return pd.DataFrame(dtype="float64")

    @staticmethod
    def _pipeline_compute_roc(data: pd.Series, period: int) -> pd.Series:
        """Compute Rate of Change (ROC) using pandas-ta.

        Args:
            data: Input Series (typically close prices).
            period: The period for ROC calculation.

        Returns:
            A Series containing the ROC values.
        """
        if not isinstance(data, pd.Series):
            return pd.Series(dtype="float64", name=f"roc_{period}")
        roc_series = data.astype("float64").ta.roc(length=period)
        if roc_series is None:
            return pd.Series(dtype="float64", name=f"roc_{period}").fillna(0.0)
        roc_series = roc_series.fillna(0.0)
        roc_series.name = f"roc_{period}"
        return roc_series.astype("float64")

    @staticmethod
    def _pipeline_compute_atr(
        ohlc_data: pd.DataFrame,
        length: int,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> pd.Series:
        """Compute Average True Range (ATR) using pandas-ta.

        Args:
            ohlc_data: DataFrame with high, low, close columns.
            length: The period for ATR calculation.
            high_col: Name of the high price column.
            low_col: Name of the low price column.
            close_col: Name of the close price column.

        Returns:
            A Series containing the ATR values.
        """
        series_name = f"atr_{length}"
        if not isinstance(ohlc_data, pd.DataFrame):
            return pd.Series(dtype="float64", name=series_name)
        if not all(col in ohlc_data.columns for col in [high_col, low_col, close_col]):
            return pd.Series(dtype="float64", name=series_name)

        atr_series = ta.atr(
            high=ohlc_data[high_col].astype("float64"),
            low=ohlc_data[low_col].astype("float64"),
            close=ohlc_data[close_col].astype("float64"),
            length=length,
        )
        if atr_series is not None:
            atr_series = atr_series.fillna(0.0)
            atr_series.name = series_name
            return atr_series.astype("float64")
        return pd.Series(dtype="float64", name=series_name).fillna(0.0)


    @staticmethod
    def _pipeline_compute_stdev(data: pd.Series, length: int) -> pd.Series:
        """Compute Standard Deviation using pandas .rolling().std().

        Args:
            data: Input Series.
            length: The rolling window length.

        Returns:
            A Series containing the standard deviation values.
        """
        if not isinstance(data, pd.Series):
            return pd.Series(dtype="float64", name=f"stdev_{length}")
        stdev_series = data.astype("float64").rolling(window=length).std()
        stdev_series = stdev_series.fillna(0.0)
        stdev_series.name = f"stdev_{length}"
        return stdev_series.astype("float64")

    @staticmethod
    def _pipeline_compute_vwap_ohlcv(
        ohlcv_df: pd.DataFrame,
        length: int,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ) -> pd.Series:
        """Compute VWAP from OHLCV data using rolling window.

        Args:
            ohlcv_df: DataFrame with price/volume columns.
            length: The rolling window length.
            high_col: Name of the high price column.
            low_col: Name of the low price column.
            close_col: Name of the close price column.
            volume_col: Name of the volume column.

        Returns:
            A Series containing the VWAP values.
        """
        series_name = f"vwap_ohlcv_{length}"
        if not isinstance(ohlcv_df, pd.DataFrame):
            return pd.Series(dtype="float64", name=series_name)
        if not all(col in ohlcv_df.columns for col in [
            high_col, low_col, close_col, volume_col,
        ]):
            return pd.Series(dtype="float64", name=series_name)

        high_d = ohlcv_df[high_col].apply(lambda x: Decimal(str(x)))
        low_d = ohlcv_df[low_col].apply(lambda x: Decimal(str(x)))
        close_d = ohlcv_df[close_col].apply(lambda x: Decimal(str(x)))
        volume_d = ohlcv_df[volume_col].apply(lambda x: Decimal(str(x)))


        typical_price = (high_d + low_d + close_d) / Decimal("3.0")
        tp_vol = typical_price * volume_d

        sum_tp_vol = tp_vol.rolling(window=length, min_periods=max(1, length)).sum()
        sum_vol = volume_d.rolling(window=length, min_periods=max(1, length)).sum()

        vwap_series_decimal = pd.Series(index=ohlcv_df.index, dtype=object)

        for idx in ohlcv_df.index:
            current_sum_tp_vol = sum_tp_vol.get(idx)
            current_sum_vol = sum_vol.get(idx)

            if pd.notna(current_sum_tp_vol) and pd.notna(current_sum_vol) and \
               current_sum_vol != Decimal("0"):
                vwap_series_decimal[idx] = current_sum_tp_vol / current_sum_vol
            elif pd.notna(high_d.get(idx)) and pd.notna(low_d.get(idx)) and \
                 pd.notna(close_d.get(idx)):
                h_val, l_val, c_val = high_d[idx], low_d[idx], close_d[idx]
                vwap_series_decimal[idx] = (h_val + l_val + c_val) / Decimal("3.0")
            else:
                vwap_series_decimal[idx] = pd.NA


        vwap_series_decimal = vwap_series_decimal.replace(
            [Decimal("Infinity"), Decimal("-Infinity")],
            pd.NA,
        )
        vwap_series_float = vwap_series_decimal.astype("float64")
        vwap_series_float.name = series_name
        return vwap_series_float

    @staticmethod
    def _pipeline_compute_vwap_trades( # noqa: PLR0912
        trade_history_deque: deque[Any],
        bar_start_times: pd.Series,
        bar_interval_seconds: int,
        ohlcv_close_prices: pd.Series | None = None,
    ) -> pd.Series:
        """Compute VWAP from trade data for specified bar start times.

        Args:
            trade_history_deque: Deque of trade dictionaries.
            bar_start_times: Series of datetime objects for bar starts.
            bar_interval_seconds: Duration of the bar in seconds.
            ohlcv_close_prices: Series of OHLCV close prices for fallback.

        Returns:
            A Series containing the trade-based VWAP values.
        """
        series_name = f"vwap_trades_{bar_interval_seconds}s"
        output_index = bar_start_times.index if isinstance(bar_start_times, pd.Series) else None
        if not isinstance(bar_start_times, pd.Series):
            return pd.Series(dtype="float64", index=output_index, name=series_name)

        vwap_results = []
        trades_df: pd.DataFrame | None = None

        if trade_history_deque:
            try:
                if not all(isinstance(trade, dict) for trade in trade_history_deque):
                    trades_df = pd.DataFrame(columns=["price", "volume", "timestamp"])
                else:
                    trades_df = pd.DataFrame(list(trade_history_deque))

                if not trades_df.empty:
                    trades_df["price"] = trades_df["price"].apply(lambda x: Decimal(str(x)))
                    trades_df["volume"] = trades_df["volume"].apply(lambda x: Decimal(str(x)))
                    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
                else:
                    trades_df = None
            except (ValueError, TypeError, KeyError, AttributeError):
                trades_df = None

        for bar_start_dt_idx, bar_start_dt in bar_start_times.items():
            calculated_vwap = np.nan
            if trades_df is not None and not trades_df.empty:
                bar_end_dt = bar_start_dt + pd.Timedelta(seconds=bar_interval_seconds)
                relevant_trades = trades_df[
                    (trades_df["timestamp"] >= bar_start_dt) &
                    (trades_df["timestamp"] < bar_end_dt)
                ]
                if not relevant_trades.empty:
                    sum_price_volume = (relevant_trades["price"] * relevant_trades["volume"]).sum()
                    sum_volume = relevant_trades["volume"].sum()
                    if sum_volume > Decimal("0"):
                        calculated_vwap = float(sum_price_volume / sum_volume)

            if pd.isna(calculated_vwap):
                current_bar_ohlcv_close_price = np.nan
                if ohlcv_close_prices is not None:
                    if bar_start_dt in ohlcv_close_prices.index:
                        current_bar_ohlcv_close_price = ohlcv_close_prices.get(bar_start_dt)
                    elif bar_start_dt_idx in ohlcv_close_prices.index:
                        current_bar_ohlcv_close_price = ohlcv_close_prices.get(bar_start_dt_idx)

                if pd.notna(current_bar_ohlcv_close_price):
                    calculated_vwap = float(current_bar_ohlcv_close_price) # type: ignore[arg-type]
                else:
                    calculated_vwap = 0.0

            vwap_results.append(calculated_vwap)

        return pd.Series(vwap_results, index=output_index, dtype="float64", name=series_name)


    @staticmethod
    def _pipeline_compute_l2_spread(l2_books_series: pd.Series) -> pd.DataFrame:
        """Compute bid-ask spread from a Series of L2 book snapshots.

        Args:
            l2_books_series: Series of L2 book snapshots.

        Returns:
            DataFrame with 'abs_spread' and 'pct_spread'.
        """
        abs_spreads = []
        pct_spreads = []
        output_index = l2_books_series.index if isinstance(l2_books_series, pd.Series) else None

        for book in l2_books_series:
            current_abs_spread = 0.0
            current_pct_spread = 0.0
            try:
                if book and \
                   isinstance(book.get("bids"), list) and len(book["bids"]) > 0 and \
                   isinstance(book["bids"][0], list | tuple) and \
                   len(book["bids"][0]) == FeatureEngine._EXPECTED_L2_LEVEL_LENGTH and \
                   isinstance(book.get("asks"), list) and len(book["asks"]) > 0 and \
                   isinstance(book["asks"][0], list | tuple) and \
                   len(book["asks"][0]) == FeatureEngine._EXPECTED_L2_LEVEL_LENGTH:

                    best_bid_price_str = str(book["bids"][0][0])
                    best_ask_price_str = str(book["asks"][0][0])

                    if not best_bid_price_str or not best_ask_price_str or \
                       best_bid_price_str == "None" or \
                       best_ask_price_str == "None":
                        raise ValueError("Empty or 'None' price string.")

                    best_bid = Decimal(best_bid_price_str)
                    best_ask = Decimal(best_ask_price_str)

                    if best_ask > best_bid:
                        abs_spread_val = best_ask - best_bid
                        mid_price = (best_bid + best_ask) / Decimal("2")
                        pct_spread_val = (abs_spread_val / mid_price) * Decimal("100") \
                            if mid_price != Decimal("0") else Decimal("0.0")
                        current_abs_spread = float(abs_spread_val)
                        current_pct_spread = float(pct_spread_val)
            except (TypeError, IndexError, ValueError, AttributeError):
                pass

            abs_spreads.append(current_abs_spread)
            pct_spreads.append(current_pct_spread)

        return pd.DataFrame(
            {"abs_spread": abs_spreads, "pct_spread": pct_spreads},
            index=output_index,
            dtype="float64",
        )

    @staticmethod
    def _pipeline_compute_l2_imbalance(
        l2_books_series: pd.Series,
        levels: int = 5,
    ) -> pd.Series:
        """Compute order book imbalance from a Series of L2 book snapshots.

        Args:
            l2_books_series: Series of L2 book snapshots.
            levels: Number of order book levels to consider.

        Returns:
            Series containing order book imbalance values.
        """
        imbalances = []
        series_name = f"imbalance_{levels}"
        output_index = l2_books_series.index if isinstance(l2_books_series, pd.Series) else None

        for book in l2_books_series:
            current_imbalance = 0.0
            try:
                if not (book and isinstance(book.get("bids"), list) and \
                        isinstance(book.get("asks"), list) and \
                        len(book["bids"]) >= levels and len(book["asks"]) >= levels):
                    raise ValueError("Invalid book structure or insufficient levels.")

                valid_levels = True
                for i in range(levels):
                    if not (
                        isinstance(book["bids"][i], list | tuple) and
                        len(book["bids"][i]) == FeatureEngine._EXPECTED_L2_LEVEL_LENGTH and
                        book["bids"][i][1] is not None and
                        isinstance(book["asks"][i], list | tuple) and
                        len(book["asks"][i]) == FeatureEngine._EXPECTED_L2_LEVEL_LENGTH and
                        book["asks"][i][1] is not None
                    ):
                        valid_levels = False
                        break
                if not valid_levels:
                    raise ValueError("Invalid level structure within L2 book.")


                bid_vol_at_levels = sum(
                    Decimal(str(book["bids"][i][1])) for i in range(levels)
                )
                ask_vol_at_levels = sum(
                    Decimal(str(book["asks"][i][1])) for i in range(levels)
                )

                total_vol = bid_vol_at_levels + ask_vol_at_levels
                if total_vol > Decimal("0"):
                    imbalance_val = (bid_vol_at_levels - ask_vol_at_levels) / total_vol
                    current_imbalance = float(imbalance_val)
            except (TypeError, IndexError, ValueError, AttributeError):
                pass

            imbalances.append(current_imbalance)

        return pd.Series(imbalances, index=output_index, dtype="float64", name=series_name)

    @staticmethod
    def _pipeline_compute_l2_wap(
        l2_books_series: pd.Series,
        ohlcv_close_prices: pd.Series | None = None,
        levels: int = 1,
    ) -> pd.Series:
        """Compute Weighted Average Price (WAP) from a Series of L2 book snapshots.

        Args:
            l2_books_series: Series of L2 book snapshots.
            ohlcv_close_prices: Series of OHLCV close prices for fallback.
            levels: Number of levels (typically 1 for WAP).

        Returns:
            Series containing WAP values.
        """
        series_name = f"wap_{levels}"
        output_index = l2_books_series.index if isinstance(l2_books_series, pd.Series) else None
        if not isinstance(l2_books_series, pd.Series):
            return pd.Series(dtype="float64", name=series_name, index=output_index)

        waps = []
        for book_idx, book in l2_books_series.items():
            calculated_wap = np.nan
            try:
                if not (book and levels >= 1 and \
                        isinstance(book.get("bids"), list) and len(book["bids"]) >= levels and \
                        isinstance(book["bids"][0], list | tuple) and \
                        len(book["bids"][0]) == FeatureEngine._EXPECTED_L2_LEVEL_LENGTH and \
                        isinstance(book.get("asks"), list) and len(book["asks"]) >= levels and \
                        isinstance(book["asks"][0], list | tuple) and \
                        len(book["asks"][0]) == FeatureEngine._EXPECTED_L2_LEVEL_LENGTH):
                    raise ValueError("Invalid book structure for WAP calculation.")


                best_bid_price_str = str(book["bids"][0][0])
                best_bid_vol_str = str(book["bids"][0][1])
                best_ask_price_str = str(book["asks"][0][0])
                best_ask_vol_str = str(book["asks"][0][1])

                if not all([
                    best_bid_price_str, best_bid_vol_str,
                    best_ask_price_str, best_ask_vol_str,
                ]) or any(s == "None" for s in [
                    best_bid_price_str, best_bid_vol_str,
                    best_ask_price_str, best_ask_vol_str,
                ]):
                    raise ValueError("Empty or 'None' price or volume string.")

                best_bid_price = Decimal(best_bid_price_str)
                best_bid_vol = Decimal(best_bid_vol_str)
                best_ask_price = Decimal(best_ask_price_str)
                best_ask_vol = Decimal(best_ask_vol_str)

                total_vol = best_bid_vol + best_ask_vol
                if total_vol > Decimal("0"):
                    wap_decimal = (
                        best_bid_price * best_ask_vol + best_ask_price * best_bid_vol
                    ) / total_vol
                    calculated_wap = float(wap_decimal)
            except (TypeError, IndexError, ValueError, AttributeError):
                pass

            if pd.isna(calculated_wap):
                if ohlcv_close_prices is not None and book_idx in ohlcv_close_prices.index:
                    fallback_close_price = ohlcv_close_prices.get(book_idx)
                    if pd.notna(fallback_close_price):
                        calculated_wap = float(fallback_close_price) # type: ignore[arg-type]
                    else:
                        calculated_wap = 0.0
                else:
                    calculated_wap = 0.0

            waps.append(calculated_wap)

        return pd.Series(waps, index=output_index, dtype="float64", name=series_name)

    @staticmethod
    def _pipeline_compute_l2_depth(
        l2_books_series: pd.Series,
        levels: int = 5,
    ) -> pd.DataFrame:
        """Compute bid and ask depth from a Series of L2 book snapshots.

        Args:
            l2_books_series: Series of L2 book snapshots.
            levels: Number of order book levels to consider.

        Returns:
            DataFrame with 'bid_depth_{levels}' and 'ask_depth_{levels}'.
        """
        bid_depths = []
        ask_depths = []
        output_index = l2_books_series.index if isinstance(l2_books_series, pd.Series) else None
        col_name_bid = f"bid_depth_{levels}"
        col_name_ask = f"ask_depth_{levels}"

        for book in l2_books_series:
            current_bid_depth = 0.0
            current_ask_depth = 0.0
            try:
                if not (book and isinstance(book.get("bids"), list) and \
                        isinstance(book.get("asks"), list) and \
                        len(book["bids"]) >= levels and len(book["asks"]) >= levels):
                    raise ValueError("Invalid book structure or insufficient levels for depth.")

                valid_levels_check = True
                for i in range(levels):
                     if not (
                        isinstance(book["bids"][i], list | tuple) and
                        len(book["bids"][i]) == FeatureEngine._EXPECTED_L2_LEVEL_LENGTH and
                        book["bids"][i][1] is not None and
                        isinstance(book["asks"][i], list | tuple) and
                        len(book["asks"][i]) == FeatureEngine._EXPECTED_L2_LEVEL_LENGTH and
                        book["asks"][i][1] is not None
                    ):
                        valid_levels_check = False
                        break
                if not valid_levels_check:
                    raise ValueError("Invalid level structure within L2 book for depth.")


                bid_depth_val = sum(
                    Decimal(str(book["bids"][i][1])) for i in range(levels)
                )
                ask_depth_val = sum(
                    Decimal(str(book["asks"][i][1])) for i in range(levels)
                )
                current_bid_depth = float(bid_depth_val)
                current_ask_depth = float(ask_depth_val)
            except (TypeError, IndexError, ValueError, AttributeError):
                pass

            bid_depths.append(current_bid_depth)
            ask_depths.append(current_ask_depth)

        return pd.DataFrame(
            {col_name_bid: bid_depths, col_name_ask: ask_depths},
            index=output_index,
            dtype="float64",
        )

    @staticmethod
    def _pipeline_compute_volume_delta(
        trade_history_deque: deque[Any],
        bar_start_times: pd.Series,
        bar_interval_seconds: int,
        ohlcv_close_prices: pd.Series | None = None, # pylint: disable=unused-argument
    ) -> pd.Series:
        """Compute Volume Delta from trade data for specified bar start times.

        Args:
            trade_history_deque: Deque of trade dictionaries.
            bar_start_times: Series of bar start datetime objects.
            bar_interval_seconds: Duration of the bar in seconds.
            ohlcv_close_prices: Series of OHLCV close prices (unused).


        Returns:
            A Series containing volume delta values.
        """
        deltas = []
        series_name = f"volume_delta_{bar_interval_seconds}s"
        output_index = bar_start_times.index if isinstance(bar_start_times, pd.Series) else None

        if not isinstance(bar_start_times, pd.Series) or \
           not isinstance(trade_history_deque, deque):
            return pd.Series(dtype="float64", index=output_index, name=series_name)

        if not trade_history_deque:
            return pd.Series(0.0, index=output_index, dtype="float64", name=series_name)

        try:
            if not all(isinstance(trade, dict) for trade in trade_history_deque):
                return pd.Series(0.0, index=output_index, dtype="float64", name=series_name)

            trades_df = pd.DataFrame(list(trade_history_deque))
            if trades_df.empty:
                 return pd.Series(0.0, index=output_index, dtype="float64", name=series_name)

            trades_df["price"] = trades_df["price"].apply(lambda x: Decimal(str(x)))
            trades_df["volume"] = trades_df["volume"].apply(lambda x: Decimal(str(x)))
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
            trades_df["side"] = trades_df["side"].astype(str).str.lower()
        except (ValueError, TypeError, KeyError, AttributeError):
            return pd.Series(0.0, index=output_index, dtype="float64", name=series_name)


        for bar_start_dt in bar_start_times:
            bar_end_dt = bar_start_dt + pd.Timedelta(seconds=bar_interval_seconds)
            relevant_trades = trades_df[
                (trades_df["timestamp"] >= bar_start_dt) &
                (trades_df["timestamp"] < bar_end_dt)
            ]

            if relevant_trades.empty:
                deltas.append(0.0)
                continue

            buy_volume = relevant_trades[relevant_trades["side"] == "buy"]["volume"].sum()
            sell_volume = relevant_trades[relevant_trades["side"] == "sell"]["volume"].sum()
            deltas.append(float(buy_volume - sell_volume))

        return pd.Series(deltas, index=output_index, dtype="float64", name=series_name)


    async def _calculate_and_publish_features( # noqa: C901, PLR0912, PLR0915
        self,
        trading_pair: str,
        timestamp_features_for: str,
    ) -> None:
        """Calculate all configured features using pipelines and publish them."""
        ohlcv_df_full_history = self.ohlcv_history.get(trading_pair)
        min_history_req = self._get_min_history_required()
        if ohlcv_df_full_history is None or len(ohlcv_df_full_history) < min_history_req:
            self.logger.info(
                "Not enough OHLCV data for %s to calculate features. Need %s, have %s.",
                trading_pair,
                min_history_req,
                len(ohlcv_df_full_history) if ohlcv_df_full_history is not None else 0,
                source_module=self._source_module,
            )
            return

        current_l2_book = self.l2_books.get(trading_pair)
        if current_l2_book and (
            not current_l2_book.get("bids") or not current_l2_book.get("asks")
        ):
            self.logger.debug(
                "L2 book for %s is present but empty or missing bids/asks. "
                "L2 features may be impacted.",
                trading_pair,
                source_module=self._source_module,
            )

        all_generated_features: dict[str, Any] = {}
        bar_start_datetime = pd.to_datetime(timestamp_features_for, utc=True)
        current_ohlcv_df_decimal = ohlcv_df_full_history[
            ohlcv_df_full_history.index <= bar_start_datetime
        ]

        if current_ohlcv_df_decimal.empty:
            self.logger.warning(
                "No historical OHLCV data for %s up to %s.",
                trading_pair,
                bar_start_datetime,
                source_module=self._source_module,
            )
            return

        close_series_for_pipelines = current_ohlcv_df_decimal["close"].astype("float64")
        ohlcv_df_for_pipelines = current_ohlcv_df_decimal.astype({
            "open": "float64", "high": "float64", "low": "float64",
            "close": "float64", "volume": "float64",
        })

        latest_l2_book_snapshot = self.l2_books.get(trading_pair)
        l2_books_aligned_series = pd.Series(
            [latest_l2_book_snapshot],
            index=[bar_start_datetime],
        )

        trades_deque = self.trade_history.get(trading_pair)
        if trades_deque is None:
            trades_deque = deque(maxlen=self.config.get("feature_engine", {}).get(
                "trade_history_maxlen",
                2000,
            ))


        bar_start_times_series = pd.Series([bar_start_datetime], index=[bar_start_datetime])

        ohlcv_close_for_dynamic_injection: pd.Series
        if bar_start_datetime in close_series_for_pipelines.index:
            ohlcv_close_for_dynamic_injection = \
                close_series_for_pipelines.loc[[bar_start_datetime]]
        else:
            self.logger.warning(
                "Could not find close price for current bar %s in historical data. "
                "Features needing this fallback may use 0.0 or NaN.",
                bar_start_datetime,
                source_module=self._source_module,
            )
            ohlcv_close_for_dynamic_injection = pd.Series(
                [np.nan],
                index=[bar_start_datetime],
                dtype="float64",
            )


        for pipeline_name, pipeline_info in self.feature_pipelines.items():
            pipeline_obj: Pipeline = pipeline_info["pipeline"]
            spec: InternalFeatureSpec = pipeline_info["spec"]
            pipeline_input_data: Any = None
            raw_pipeline_output: Any = None

            if spec.input_type == "close_series":
                pipeline_input_data = close_series_for_pipelines
            elif spec.input_type == "ohlcv_df":
                pipeline_input_data = ohlcv_df_for_pipelines
            elif spec.input_type == "l2_book_series":
                pipeline_input_data = l2_books_aligned_series
            elif spec.input_type == "trades_and_bar_starts":
                pipeline_input_data = trades_deque
            else:
                self.logger.warning(
                    "Unknown input_type '%s' for pipeline %s. Skipping.",
                    spec.input_type,
                    pipeline_name,
                    source_module=self._source_module,
                )
                continue

            try:
                pipeline_to_run = pipeline_obj
                if spec.calculator_type in ["l2_wap", "vwap_trades", "volume_delta"]:
                    pipeline_to_run = clone(pipeline_obj)
                    calculator_step_name = f"{spec.key}_calculator"
                    if calculator_step_name in pipeline_to_run.named_steps:
                        calc_transformer = pipeline_to_run.named_steps[calculator_step_name]
                        if hasattr(calc_transformer, "kw_args"):
                            current_kw_args = calc_transformer.kw_args.copy()
                            current_kw_args["ohlcv_close_prices"] = \
                                ohlcv_close_for_dynamic_injection
                            if spec.input_type == "trades_and_bar_starts":
                                current_kw_args["bar_start_times"] = bar_start_times_series
                            calc_transformer.kw_args = current_kw_args
                        else:
                            self.logger.error(
                                "Calculator step %s in pipeline %s does not have kw_args.",
                                calculator_step_name, pipeline_name,
                                source_module=self._source_module,
                            )
                    else:
                        self.logger.error(
                            "Calculator step %s not found in cloned pipeline %s.",
                            calculator_step_name,
                            pipeline_name,
                            source_module=self._source_module,
                        )


                if pipeline_input_data is not None:
                    if (isinstance(pipeline_input_data, pd.Series | pd.DataFrame) \
                        and pipeline_input_data.empty) or \
                       (isinstance(pipeline_input_data, deque) and not pipeline_input_data):
                        self.logger.debug(
                            "Pipeline input data for %s is empty. Skipping execution.",
                            pipeline_name,
                            source_module=self._source_module,
                        )
                        continue
                    raw_pipeline_output = pipeline_to_run.fit_transform(pipeline_input_data)
                else:
                    self.logger.warning(
                        "Pipeline input data is None for %s. This should not happen.",
                        pipeline_name,
                        source_module=self._source_module,
                    )
                    continue

            except Exception as e:
                self.logger.exception(
                    "Error executing pipeline %s: %s",
                    pipeline_name,
                    e,
                    source_module=self._source_module,
                )
                continue

            latest_features_values: Any = None
            if isinstance(raw_pipeline_output, pd.Series):
                if not raw_pipeline_output.empty:
                    if spec.input_type not in ["l2_book_series", "trades_and_bar_starts"] \
                       or len(raw_pipeline_output) > 1:
                        latest_features_values = raw_pipeline_output.iloc[-1]
                    else:
                        latest_features_values = raw_pipeline_output.iloc[0] \
                            if len(raw_pipeline_output) == 1 else np.nan
            elif isinstance(raw_pipeline_output, pd.DataFrame):
                if not raw_pipeline_output.empty:
                    if spec.input_type not in ["l2_book_series"] \
                       or len(raw_pipeline_output) > 1:
                        latest_features_values = raw_pipeline_output.iloc[-1]
                    else:
                        latest_features_values = raw_pipeline_output.iloc[0] \
                            if len(raw_pipeline_output) == 1 else pd.Series(dtype="float64")
            elif isinstance(raw_pipeline_output, np.ndarray):
                if raw_pipeline_output.ndim == NUMPY_1D_ARRAY_DIM and \
                   raw_pipeline_output.size > 0:
                    latest_features_values = raw_pipeline_output[-1]
                elif raw_pipeline_output.ndim == NUMPY_2D_ARRAY_DIM and \
                     raw_pipeline_output.shape[0] > 0:
                    latest_features_values = pd.Series(raw_pipeline_output[-1, :])
            else:
                latest_features_values = raw_pipeline_output


            if isinstance(latest_features_values, pd.Series):
                for idx_name, value in latest_features_values.items():
                    col_name = str(idx_name)
                    base_feature_key = spec.key
                    feature_output_name = f"{base_feature_key}_{col_name}"
                    all_generated_features[feature_output_name] = float(value) \
                        if pd.notna(value) else np.nan
            elif pd.notna(latest_features_values):
                feature_output_name = spec.key
                all_generated_features[feature_output_name] = float(latest_features_values) \
                    if pd.notna(latest_features_values) else np.nan


        try:
            features_for_pydantic: dict[str, float] = {}
            for k, v_any in all_generated_features.items():
                if pd.notna(v_any) and isinstance(v_any, float | int | np.number):
                    features_for_pydantic[k] = float(v_any)
                else:
                    features_for_pydantic[k] = np.nan


            pydantic_features = PublishedFeaturesV1(**features_for_pydantic)
            features_for_payload = pydantic_features.model_dump()
        except Exception as e:
            self.logger.error(
                "Failed to validate/structure features for %s at %s: %s. Raw features: %s",
                trading_pair,
                timestamp_features_for,
                e,
                all_generated_features,
                source_module=self._source_module,
            )
            return

        if not features_for_payload:
            self.logger.info(
                "No features were structured/validated for %s at %s. Not publishing.",
                trading_pair,
                timestamp_features_for,
                source_module=self._source_module,
            )
            return

        event_payload = {
            "trading_pair": trading_pair,
            "exchange": self.config.get("exchange_name", "unknown_exchange"),
            "timestamp_features_for": timestamp_features_for,
            "features": features_for_payload,
        }
        full_feature_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": EventType.FEATURES_CALCULATED.name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_module": self._source_module,
            "payload": event_payload,
        }

        try:
            from typing import cast
            await self.pubsub_manager.publish(cast("Any", full_feature_event))
            self.logger.info(
                "Published FEATURES_CALCULATED event for %s at %s",
                trading_pair,
                timestamp_features_for,
                source_module=self._source_module,
                context={
                    "event_id": full_feature_event["event_id"],
                    "num_features": len(features_for_payload),
                },
            )
        except Exception as e:
            self.logger.exception(
                "Failed to publish FEATURES_CALCULATED event for %s: %s",
                trading_pair,
                str(e),
                source_module=self._source_module,
            )

    def _format_feature_value(self, value: Decimal | float | object) -> str: # pylint: disable=unused-private-member
        """Format a feature value to string. Decimal/float to 8 decimal places.

        Args:
            value: The value to format.

        Returns:
            The formatted string representation of the value.
        """
        if isinstance(value, Decimal | float):
            return f"{value:.8f}"
        return str(value)
