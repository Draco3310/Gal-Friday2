"""Unit tests for FeatureEngine._build_feature_pipelines."""

from unittest.mock import MagicMock

import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler

from gal_friday.feature_engine import FeatureEngine, InternalFeatureSpec, PandasScalerTransformer

# --- Mocks and Helpers ---

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def mock_pubsub_manager():
    return MagicMock()

def create_feature_engine_with_config(
    feature_config: dict,
    logger: MagicMock,
    pubsub_manager: MagicMock,
) -> FeatureEngine:
    """Helper to instantiate FeatureEngine with a specific feature configuration."""
    base_config = {
        "exchange_name": "test_exchange",
        "feature_engine": {"trade_history_maxlen": 100},
        "features": feature_config,
    }
    engine = FeatureEngine(config=base_config, pubsub_manager=pubsub_manager, logger_service=logger)
    return engine

# --- Test Cases ---

def test_build_rsi_pipeline_custom_config(mock_logger, mock_pubsub_manager):
    """Test RSI pipeline construction with full custom configuration."""
    feature_key = "rsi_14_custom"
    rsi_config = {
        feature_key: {
            "calculator_type": "rsi",
            "input_type": "close_series",
            "category": "TECHNICAL",
            "parameters": {"period": 14},
            "imputation": {"strategy": "constant", "fill_value": 50.0},
            "scaling": {"method": "minmax", "feature_range": (0, 1)},
        },
    }
    engine = create_feature_engine_with_config(rsi_config, mock_logger, mock_pubsub_manager)

    pipeline_name = f"{feature_key}_pipeline"
    assert pipeline_name in engine.feature_pipelines
    pipeline_info = engine.feature_pipelines[pipeline_name]
    pipeline = pipeline_info["pipeline"]
    spec = pipeline_info["spec"]

    assert isinstance(pipeline, Pipeline)
    assert isinstance(spec, InternalFeatureSpec)
    assert spec.key == feature_key
    assert spec.parameters["period"] == 14

    # Expected steps: input_imputer, rsi_calculator, reshape_before_impute, output_imputer, reshape_after_impute, output_scaler
    # Check number of steps (can be fragile if reshape steps are conditional)
    # For constant SimpleImputer, it should be: input_imputer, calculator, reshape_before, SimpleImputer, reshape_after, scaler
    # For fillna func transformer: input_imputer, calculator, fillna_transformer, scaler

    # Assuming the current logic:
    # 0: input_imputer (SimpleImputer for close series)
    # 1: calculator (FunctionTransformer for _pipeline_compute_rsi)
    # 2: reshape_before_impute (FunctionTransformer)
    # 3: output_imputer (SimpleImputer for RSI output)
    # 4: reshape_after_impute (FunctionTransformer)
    # 5: output_scaler (PandasScalerTransformer with MinMaxScaler)

    assert pipeline.steps[0][0] == f"{feature_key}_input_imputer" # Name based on spec.key
    assert isinstance(pipeline.steps[0][1], SimpleImputer)
    assert pipeline.steps[0][1].strategy == "mean" # Default for input

    assert pipeline.steps[1][0] == f"{feature_key}_calculator"
    assert isinstance(pipeline.steps[1][1], FunctionTransformer)
    assert pipeline.steps[1][1].kw_args == {"period": 14}

    assert pipeline.steps[3][0] == f"{feature_key}_output_imputer_const_50.0" # Name from helper
    assert isinstance(pipeline.steps[3][1], SimpleImputer) # If using SimpleImputer path
    assert pipeline.steps[3][1].strategy == "constant"
    assert pipeline.steps[3][1].fill_value == 50.0

    assert pipeline.steps[5][0] == f"{feature_key}_output_scaler_MinMaxScaler" # Name from helper
    assert isinstance(pipeline.steps[5][1], PandasScalerTransformer)
    assert isinstance(pipeline.steps[5][1].scaler, MinMaxScaler)
    assert pipeline.steps[5][1].scaler.feature_range == (0, 1)

def test_build_rsi_pipeline_default_processing(mock_logger, mock_pubsub_manager):
    """Test RSI pipeline with default output imputation and scaling."""
    feature_key = "rsi_20_default"
    rsi_config = {
        feature_key: {
            "calculator_type": "rsi", "input_type": "close_series",
            "parameters": {"period": 20},
            "imputation": None, # Trigger default output imputation (fillna 50)
            "scaling": None,     # Trigger default output scaling (StandardScaler)
        },
    }
    engine = create_feature_engine_with_config(rsi_config, mock_logger, mock_pubsub_manager)
    pipeline_name = f"{feature_key}_pipeline"
    assert pipeline_name in engine.feature_pipelines
    pipeline = engine.feature_pipelines[pipeline_name]["pipeline"]

    # Expected: input_imputer, calculator, output_fillna_50_transformer, output_scaler (Standard)
    assert pipeline.steps[2][0] == f"{feature_key}_output_fillna" # Default fillna for rsi
    assert isinstance(pipeline.steps[2][1], FunctionTransformer) # Default fillna for RSI is 50

    assert pipeline.steps[3][0] == f"{feature_key}_output_scaler_StandardScaler"
    assert isinstance(pipeline.steps[3][1], PandasScalerTransformer)
    assert isinstance(pipeline.steps[3][1].scaler, StandardScaler)

def test_build_pipeline_passthrough_processing(mock_logger, mock_pubsub_manager):
    """Test 'passthrough' for imputation and scaling."""
    feature_key = "rsi_passthrough"
    rsi_config = {
        feature_key: {
            "calculator_type": "rsi", "input_type": "close_series",
            "parameters": {"period": 7},
            "imputation": "passthrough",
            "scaling": "passthrough",
        },
    }
    engine = create_feature_engine_with_config(rsi_config, mock_logger, mock_pubsub_manager)
    pipeline_name = f"{feature_key}_pipeline"
    assert pipeline_name in engine.feature_pipelines
    pipeline = engine.feature_pipelines[pipeline_name]["pipeline"]
    # Expected: input_imputer, calculator ONLY
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == f"{feature_key}_input_imputer"
    assert pipeline.steps[1][0] == f"{feature_key}_calculator"


def test_build_macd_pipeline(mock_logger, mock_pubsub_manager):
    feature_key = "macd_custom"
    macd_config = {
        feature_key: {
            "calculator_type": "macd", "input_type": "close_series",
            "parameters": {"fast": 10, "slow": 20, "signal": 5},
            "imputation": {"strategy": "constant", "fill_value": 0.1}, # For DataFrame
            "scaling": {"method": "robust"},
        },
    }
    engine = create_feature_engine_with_config(macd_config, mock_logger, mock_pubsub_manager)
    pipeline_name = f"{feature_key}_pipeline"
    assert pipeline_name in engine.feature_pipelines
    pipeline = engine.feature_pipelines[pipeline_name]["pipeline"]

    assert pipeline.steps[1][1].kw_args == {"fast": 10, "slow": 20, "signal": 5}
    assert pipeline.steps[2][0] == f"{feature_key}_output_imputer_const_0.1" # DataFrame fillna
    assert isinstance(pipeline.steps[3][1].scaler, RobustScaler)


def test_build_l2_spread_pipeline(mock_logger, mock_pubsub_manager):
    """Test L2 Spread (DataFrame output, no input imputer in this simple pipeline)."""
    feature_key = "l2_spread_test"
    l2_config = {
        feature_key: {
            "calculator_type": "l2_spread", "input_type": "l2_book_series",
            # No specific calc params for spread
            "imputation": None, # Default: df.fillna(df.mean())
            "scaling": "passthrough",
        },
    }
    engine = create_feature_engine_with_config(l2_config, mock_logger, mock_pubsub_manager)
    pipeline_name = f"{feature_key}_pipeline"
    assert pipeline_name in engine.feature_pipelines
    pipeline = engine.feature_pipelines[pipeline_name]["pipeline"]

    # Expected: calculator, output_imputer (df.fillna(df.mean()))
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == f"{feature_key}_calculator"
    assert pipeline.steps[1][0] == f"{feature_key}_output_fillna" # Default for DataFrame

def test_build_volume_delta_pipeline(mock_logger, mock_pubsub_manager):
    """Test Volume Delta (special input, Series output)."""
    feature_key = "vol_delta_test"
    vd_config = {
        feature_key: {
            "calculator_type": "volume_delta", "input_type": "trades_and_bar_starts",
            "parameters": {"bar_interval_seconds": 120},
            "imputation": {"strategy": "constant", "fill_value": -1.0},
            "scaling": {"method": "standard"},
        },
    }
    engine = create_feature_engine_with_config(vd_config, mock_logger, mock_pubsub_manager)
    pipeline_name = f"{feature_key}_pipeline"
    assert pipeline_name in engine.feature_pipelines
    pipeline = engine.feature_pipelines[pipeline_name]["pipeline"]
    spec = engine.feature_pipelines[pipeline_name]["spec"]

    assert spec.input_type == "trades_and_bar_starts"
    # Expected: calculator, output_imputer, output_scaler
    assert len(pipeline.steps) == 3
    assert pipeline.steps[0][0] == f"{feature_key}_calculator"
    assert pipeline.steps[0][1].kw_args == {"bar_interval_seconds": 120}

    assert pipeline.steps[1][0] == f"{feature_key}_output_imputer_const_-1.0"
    assert isinstance(pipeline.steps[2][1].scaler, StandardScaler)

def test_missing_critical_parameter(mock_logger, mock_pubsub_manager):
    """Test that a feature pipeline might not be built or logs error if critical param is missing."""
    feature_key = "rsi_no_period"
    rsi_config = {
        feature_key: {
            "calculator_type": "rsi", "input_type": "close_series",
            # "parameters": {"period": 10}, # Period is missing
        },
    }
    engine = create_feature_engine_with_config(rsi_config, mock_logger, mock_pubsub_manager)
    # Default period is 14 for rsi, so it should still build.
    # _build_feature_pipelines logs a debug message if default is used.
    pipeline_name = f"{feature_key}_pipeline"
    assert pipeline_name in engine.feature_pipelines
    assert engine.feature_pipelines[pipeline_name]["pipeline"].steps[1][1].kw_args["period"] == 14
    # Check if logger was called with a debug message (or warning if we change it)
    # This requires more advanced mock inspection, e.g. mock_logger.debug.assert_any_call(...)
    # For now, we confirm default is applied.

    # Example: A feature type where a param is absolutely essential and has no default
    feature_key_bad = "my_custom_feature_no_param"
    custom_config = {
         feature_key_bad: {
            "calculator_type": "non_existent_type_for_this_test", # Will cause calc_func to be None
            "input_type": "close_series",
        },
    }
    engine_bad = create_feature_engine_with_config(custom_config, mock_logger, mock_pubsub_manager)
    assert f"{feature_key_bad}_pipeline" not in engine_bad.feature_pipelines
    # Check for error log: mock_logger.error.assert_called_with(...) containing "No _pipeline_compute function"
    # This can be done by inspecting mock_logger.error.call_args_list in a real test runner.

    # Verify one of the logger calls if needed, e.g.
    # log_calls = [call[0][0] for call in mock_logger.error.call_args_list]
    # assert any("No _pipeline_compute function found" in call_str for call_str in log_calls)
