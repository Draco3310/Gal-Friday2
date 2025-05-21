#!/usr/bin/env python
"""
Script to train the initial prediction model (MVP - XGBoost).

Reads historical data, generates features and labels based on configuration,
trains an XGBoost model, evaluates it, and saves the model artifact.
"""

import logging
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost as xgb

# Error messages and constants
FEATURE_LIST_ERROR = "Feature list ('training.data.feature_list') not defined in config"
SPLIT_RATIO_ERROR = "Train split ratio %s results in empty train or test set"
MODEL_TYPE_ERROR = "Model type in config is '%s', but this script trains XGBoost"
DATA_PATH_ERROR = "Data path not found: %s"
NO_TRADING_PAIRS_ERROR = "No trading pairs configured."
NO_TIMESTAMP_COL_ERROR = "Cannot find a suitable timestamp column."
INVALID_INDEX_ERROR = "Expected DatetimeIndex but got different index type"
NO_DATA_FOR_PAIR_ERROR = "No data found for pair %s in %s"
HISTORICAL_DATA_PATH_ERROR = "Historical data path not found or configured: %s"
LOADING_PAIR_MSG = "Loading data for training pair: %s from %s"
MISSING_FEATURES_ERROR = "Missing required features: %s"
LABELING_CONFIG_ERROR = (
    "Labeling horizon and threshold must be set in config ('training.labeling')"
)
MIN_CLASSES_REQUIRED = 2  # Minimum number of classes required for binary classification

# Error messages
MODEL_PATH_ERROR = "Model save path ('prediction_service.model_path') not configured"

# Add src directory to Python path to allow importing modules like ConfigManager
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from gal_friday.config_manager import ConfigManager
except ImportError as e:
    print(f"Error importing ConfigManager: {e}")
    print("Ensure the script is run from the project root or the PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Logging Setup --- #
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
log = logging.getLogger(__name__)

# --- Helper Functions --- #


def _handle_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    """Handle timezone-aware timestamps in the DataFrame.

    Args:
        df: Input DataFrame

    Returns
    -------
        DataFrame with properly formatted DatetimeIndex

    Raises
    ------
        ValueError: If no suitable timestamp column is found
        TypeError: If the index is not a DatetimeIndex
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        timestamp_cols = ["timestamp", "time", "date"]
        found_col = next((col for col in timestamp_cols if col in df.columns), None)

        if found_col:
            df[found_col] = pd.to_datetime(df[found_col])
            if df[found_col].dt.tz is not None:
                df[found_col] = df[found_col].dt.tz_convert("UTC")
            return df.set_index(found_col)
        raise ValueError(NO_TIMESTAMP_COL_ERROR)

    if df.index.tz is None:
        return df.tz_localize("UTC")

    if isinstance(df.index, pd.DatetimeIndex):
        return df.tz_convert("UTC")

    raise TypeError(INVALID_INDEX_ERROR)


def _filter_and_clean_data(
    all_data: pd.DataFrame,
    pair_to_train: str,
    data_path: str
) -> pd.DataFrame:
    """Filter data for target pair and clean it.

    Args:
        all_data: Input DataFrame with all pairs
        pair_to_train: Target trading pair
        data_path: Path to the data file (for error messages)

    Returns
    -------
        Cleaned DataFrame with data for the target pair
    """
    log = logging.getLogger(__name__)

    # Filter for the target pair
    df = all_data[all_data["pair"] == pair_to_train].copy()
    log.info("Loaded %d rows for %s.", len(df), pair_to_train)

    if df.empty:
        error_msg = NO_DATA_FOR_PAIR_ERROR % (pair_to_train, data_path)
        log.error(error_msg)
        raise ValueError(error_msg)

    # Ensure numeric types for OHLCV (needed for pandas_ta)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any rows with missing OHLCV data
    cleaned_df = df.dropna(
        subset=["open", "high", "low", "close", "volume"],
        inplace=False
    )
    log.info("%d rows remaining after initial NaN drop.", len(cleaned_df))
    return cleaned_df


def load_historical_data(config: "ConfigManager") -> pd.DataFrame:
    """Load and prepare historical data for a single pair.

    Args:
        config: ConfigManager instance with application configuration

    Returns
    -------
        pd.DataFrame: DataFrame with historical data for the specified pair

    Raises
    ------
        FileNotFoundError: If data file is not found
        ValueError: If no trading pairs are configured or data is invalid
        TypeError: If the data index is not a DatetimeIndex
    """
    log = logging.getLogger(__name__)
    data_path = config.get("backtest.data_path")
    trading_pairs = config.get_list("trading.pairs")

    data_path_obj = Path(data_path) if data_path else None
    if not data_path or not data_path_obj.exists():
        log.error(HISTORICAL_DATA_PATH_ERROR, data_path)
        raise FileNotFoundError(DATA_PATH_ERROR % data_path)

    if not trading_pairs:
        log.error("No trading pairs configured ('trading.pairs').")
        raise ValueError(NO_TRADING_PAIRS_ERROR)

    # For the initial script, focus on the first configured pair
    pair_to_train = trading_pairs[0]
    log.info(LOADING_PAIR_MSG, pair_to_train, data_path)

    try:
        all_data = pd.read_parquet(data_path)
        all_data = _handle_timestamp_index(all_data)
        all_data = all_data.sort_index()

        return _filter_and_clean_data(all_data, pair_to_train, data_path)
    except Exception:
        log.exception("Error loading historical data")
        raise


def generate_features(df: pd.DataFrame, config: "ConfigManager") -> pd.DataFrame:
    """Generate features based on configuration.

    Args
    ----
        df: Input DataFrame containing raw OHLCV data
        config: Configuration manager instance containing feature settings

    Returns
    -------
        DataFrame with additional technical analysis features
    """
    log.info("Generating features...")
    feature_config = config.get("feature_engine", {})
    if not feature_config:
        log.warning("No 'feature_engine' configuration found. Skipping feature generation.")
        return df

    enabled_features = feature_config.get("enabled_features", [])
    log.info("Enabled features: %s", enabled_features)

    # Example using pandas_ta based on config
    if "rsi" in enabled_features:
        rsi_period = feature_config.get("rsi_period", 14)
        df.ta.rsi(length=rsi_period, append=True)
        log.debug("Calculated RSI(%d)", rsi_period)

    if "macd" in enabled_features:
        macd_fast = feature_config.get("macd_fast_period", 12)
        macd_slow = feature_config.get("macd_slow_period", 26)
        macd_signal = feature_config.get("macd_signal_period", 9)
        df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)
        log.debug(
            "Calculated MACD(fast=%d,slow=%d,signal=%d)",
            macd_fast,
            macd_slow,
            macd_signal,
        )

    if "bbands" in enabled_features:
        bb_len = feature_config.get("bbands_period", 20)
        bb_std = feature_config.get("bbands_std_dev", 2)
        df.ta.bbands(length=bb_len, std=bb_std, append=True)
        log.debug("Calculated BBands(length=%d,std=%.1f)", bb_len, bb_std)

    if "atr" in enabled_features:
        atr_period = config.get_int(
            "backtest.atr_period", 14
        )  # Reuse from backtest for consistency
        df.ta.atr(length=atr_period, append=True)
        log.debug("Calculated ATR(%d)", atr_period)

    # Add more feature calculations here based on config...

    # Clean up column names generated by pandas_ta (e.g., RSI_14, MACDh_12_26_9)
    # Fix for mypy error: convert to list first then assign back
    columns_list = [col.lower() for col in df.columns]
    df.columns = pd.Index(columns_list)

    # Select only the features specified in the training config
    feature_list = config.get_list("training.data.feature_list", [])
    base_features = ["open", "high", "low", "close", "volume"]
    feature_list = base_features + feature_list

    missing_features = [f for f in feature_list if f not in df.columns]
    if missing_features:
        log.error("Configured features not generated: %s", missing_features)
        log.error("Available columns after generation: %s", df.columns.tolist())
        raise ValueError(MISSING_FEATURES_ERROR % missing_features)

    # Drop rows with NaNs introduced by feature calculation
    initial_rows = len(df)
    df.dropna(inplace=True)
    log.info(
        "Dropped %d rows due to NaNs after feature generation.",
        initial_rows - len(df),
    )
    log.info("Feature generation complete. %d rows remaining.", len(df))

    return df


def generate_labels(df: pd.DataFrame, config: "ConfigManager") -> pd.DataFrame:
    """Generate the target variable (label) based on future price movement.

    Args
    ----
        df: Input DataFrame containing OHLCV and feature data
        config: Configuration manager instance containing labeling parameters

    Returns
    -------
        DataFrame with added target labels
    """
    log.info("Generating labels...")
    label_config = config.get("training.labeling", {})
    horizon_minutes = label_config.get("target_horizon_minutes")
    threshold_pct = label_config.get("target_threshold_pct")

    if horizon_minutes is None or threshold_pct is None:
        raise ValueError(LABELING_CONFIG_ERROR)

    log.info(
        "Labeling parameters: Horizon=%d mins, Threshold=%.2f%%",
        horizon_minutes,
        threshold_pct,
    )

    # Ensure we have minutely data or adjust logic
    # This assumes 1-minute frequency based on typical OHLCV
    horizon_periods = horizon_minutes

    # Calculate future close price
    df["future_close"] = df["close"].shift(-horizon_periods)

    # Calculate percentage change
    df["future_pct_change"] = (df["future_close"] - df["close"]) / df["close"] * 100

    # Create binary target label
    df["target"] = (df["future_pct_change"] >= threshold_pct).astype(int)

    # Drop rows where future data is unavailable (introduced by shift)
    initial_rows = len(df)
    df.dropna(subset=["future_close", "target"], inplace=True)
    log.info(
        "Dropped %d rows due to NaNs after label generation.",
        initial_rows - len(df),
    )

    # Log label distribution
    label_counts = df["target"].value_counts(normalize=True) * 100
    log.info("Label distribution:")
    log.info("%s", label_counts)

    if len(label_counts) < MIN_CLASSES_REQUIRED:
        log.warning(
            "Dataset contains only one class after labeling. "
            "Model training might fail or be meaningless."
        )

    log.info("Label generation complete. %d rows remaining.", len(df))
    return df.drop(columns=["future_close", "future_pct_change"])  # Clean up intermediate columns


def split_data(
    df: pd.DataFrame, config: "ConfigManager"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data chronologically into training and testing sets.

    Args
    ----
        df: Input DataFrame containing features and labels
        config: Configuration manager instance containing split parameters

    Returns
    -------
        Tuple containing (X_train, X_test, y_train, y_test)
    """
    log.info("Splitting data...")
    feature_list = config.get_list("training.data.feature_list")
    split_ratio = config.get("training.data.train_split_ratio", 0.8)

    if not feature_list:
        raise ValueError(FEATURE_LIST_ERROR)

    x_data = df[feature_list]
    y_data = df["target"]

    split_index = int(len(df) * split_ratio)
    if split_index == 0 or split_index == len(df):
        raise ValueError(SPLIT_RATIO_ERROR % split_ratio)

    x_train = x_data.iloc[:split_index]
    x_test = x_data.iloc[split_index:]
    y_train = y_data.iloc[:split_index]
    y_test = y_data.iloc[split_index:]

    log.info(
        "Data split: Train=%d rows, Test=%d rows",
        len(x_train),
        len(x_test)
    )
    train_dist = y_train.value_counts(normalize=True) * 100
    log.info("Training Target Distribution:\n%s", train_dist)
    test_dist = y_test.value_counts(normalize=True) * 100
    log.info("Testing Target Distribution:\n%s", test_dist)

    return x_train, x_test, y_train, y_test


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    config: "ConfigManager",
) -> xgb.XGBClassifier:
    """Train an XGBoost model with the provided data.

    Args
    ----
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: Configuration manager instance containing model parameters

    Returns
    -------
        Trained XGBoost classifier model
    """
    log.info("Training XGBoost model...")
    model_config = config.get("training.model", {})
    model_params = model_config.get("params", {})
    model_type = model_config.get("type")

    if model_type != "xgboost":
        raise ValueError(MODEL_TYPE_ERROR % model_type)

    if not model_params:
        log.warning(
            "No model parameters ('training.model.params') found in config. "
            "Using XGBoost defaults."
        )

    # Check target balance and set scale_pos_weight if needed
    balance = y_train.value_counts()
    if 0 in balance and 1 in balance and balance[0] > 0:
        scale_pos_weight = balance[0] / balance[1]
        log.info(
            "Target imbalance detected. Setting scale_pos_weight=%.2f",
            scale_pos_weight
        )
        model_params["scale_pos_weight"] = scale_pos_weight
    else:
        log.warning("Could not calculate scale_pos_weight (single class or division by zero).")

    model = xgb.XGBClassifier(**model_params, use_label_encoder=False)
    # Pass X_test, y_test to eval_set requires them to be passed into
    # this function. Simplified: fit only on train data for now
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        verbose=True
    )

    log.info("Model training complete.")
    return model


def evaluate_model(model: xgb.XGBClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Evaluate model performance on test data.

    Args
    ----
        model: Trained XGBoost classifier
        x_test: Test features
        y_test: Test labels
    """
    log.info("Evaluating model on test set...")
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]  # Probability of positive class

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    log.info("Test Set Performance:")
    log.info("  Accuracy:  %.4f", accuracy)
    log.info("  Precision: %.4f", precision)
    log.info("  Recall:    %.4f", recall)
    log.info("  F1-Score:  %.4f", f1)
    log.info("  ROC AUC:   %.4f", roc_auc)


def save_model(model: xgb.XGBClassifier, config: "ConfigManager") -> None:
    """Save the trained model to disk.

    Args
    ----
        model: Trained XGBoost classifier to save
        config: Configuration manager instance containing save path
    """
    save_path = config.get("prediction_service.model_path")
    if not save_path:
        raise ValueError(MODEL_PATH_ERROR)

    # Ensure the directory exists
    save_path = Path(save_path)
    model_dir = save_path.parent
    if not model_dir.exists():
        log.info("Creating model directory: %s", model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

    log.info("Saving trained model to: %s", save_path)
    try:
        joblib.dump(model, save_path)
        log.info("Model saved successfully.")
    except Exception:
        log.exception("Error saving model to %s", save_path)
        raise


# --- Main Execution --- #
def main() -> None:
    """Execute the initial model training pipeline.

    Loads data, generates features, creates labels, trains and evaluates
    an XGBoost model for price movement prediction.
    """
    log.info("Starting initial model training script...")
    try:
        config = ConfigManager(config_path="config/config.yaml")

        # 1. Load Data
        df_raw = load_historical_data(config)

        # 2. Generate Features
        df_features = generate_features(df_raw, config)

        # 3. Generate Labels
        df_labeled = generate_labels(df_features, config)

        # 4. Split Data
        x_train, x_test, y_train, y_test = split_data(df_labeled, config)

        # 5. Train Model
        # Pass test data for evaluation during training if desired
        model = train_model(x_train, y_train, x_test, y_test, config)  # Pass x_test, y_test

        # 6. Evaluate Model
        evaluate_model(model, x_test, y_test)

        # 7. Save Model
        save_model(model, config)

        log.info("Initial model training script finished successfully.")

    except FileNotFoundError:
        log.exception("Configuration or data file not found")
        raise
    except ValueError:
        log.exception("Configuration error or data issue")
        raise
    except Exception:
        log.exception("An unexpected error occurred during training")
        raise


if __name__ == "__main__":
    main()
