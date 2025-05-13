#!/usr/bin/env python
"""
Script to train the initial prediction model (MVP - XGBoost).

Reads historical data, generates features and labels based on configuration,
 trains an XGBoost model, evaluates it, and saves the model artifact.
"""

import logging
import os
import sys
from typing import Tuple

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Add src directory to Python path to allow importing modules like ConfigManager
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

try:
    from gal_friday.config_manager import ConfigManager  # type: ignore
except ImportError as e:
    print(f"Error importing ConfigManager: {e}")
    print("Ensure the script is run from the project root or the PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Logging Setup --- #
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
log = logging.getLogger(__name__)

# --- Helper Functions --- #


def load_historical_data(config: "ConfigManager") -> pd.DataFrame:
    """Load and prepare historical data for a single pair.

    Args:
        config: Configuration manager instance containing data paths and trading pairs

    Returns:
        DataFrame containing cleaned and prepared historical data
    """
    data_path = config.get("backtest.data_path")
    trading_pairs = config.get_list("trading.pairs")

    if not data_path or not os.path.exists(data_path):
        log.error(f"Historical data path not found or not configured: {data_path}")
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if not trading_pairs:
        log.error("No trading pairs configured ('trading.pairs').")
        raise ValueError("No trading pairs configured.")

    # For the initial script, focus on the first configured pair
    pair_to_train = trading_pairs[0]
    log.info(f"Loading data for training pair: {pair_to_train} from {data_path}")

    try:
        all_data = pd.read_parquet(data_path)

        # Basic Cleaning (similar to backtester)
        if not isinstance(all_data.index, pd.DatetimeIndex):
            ts_cols = ["timestamp", "time", "date"]
            found_col = next((col for col in ts_cols if col in all_data.columns), None)
            if found_col:
                all_data[found_col] = pd.to_datetime(all_data[found_col])
                if all_data[found_col].dt.tz is None:
                    all_data[found_col] = all_data[found_col].dt.tz_localize("UTC")
                else:
                    all_data[found_col] = all_data[found_col].dt.tz_convert("UTC")
                all_data = all_data.set_index(found_col)
            else:
                raise ValueError("Cannot find a suitable timestamp column.")
        elif all_data.index.tz is None:
            all_data = all_data.tz_localize("UTC")
        else:
            # For mypy, ensure we're operating on a DatetimeIndex
            if isinstance(all_data.index, pd.DatetimeIndex):
                all_data.index = all_data.index.tz_convert("UTC")
            else:
                raise TypeError("Expected DatetimeIndex but got different index type")

        all_data = all_data.sort_index()

        # Filter for the target pair
        df = all_data[all_data["pair"] == pair_to_train].copy()
        log.info(f"Loaded {len(df)} rows for {pair_to_train}.")

        if df.empty:
            raise ValueError(f"No data found for pair {pair_to_train} in {data_path}")

        # Ensure numeric types for OHLCV (needed for pandas_ta)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Explicitly type the DataFrame after dropna to assist mypy
        cleaned_df: pd.DataFrame = df.dropna(
            subset=["open", "high", "low", "close", "volume"], inplace=False
        )
        log.info(f"{len(cleaned_df)} rows remaining after initial NaN drop.")

        return cleaned_df

    except Exception as e:
        log.exception(f"Error loading historical data: {e}", exc_info=True)
        raise


def generate_features(df: pd.DataFrame, config: "ConfigManager") -> pd.DataFrame:
    """Generate features based on configuration.

    Args:
        df: Input DataFrame containing raw OHLCV data
        config: Configuration manager instance containing feature settings

    Returns:
        DataFrame with additional technical analysis features
    """
    log.info("Generating features...")
    feature_config = config.get("feature_engine", {})
    if not feature_config:
        log.warning("No 'feature_engine' configuration found. Skipping feature generation.")
        return df

    enabled_features = feature_config.get("enabled_features", [])
    log.info(f"Enabled features: {enabled_features}")

    # Example using pandas_ta based on config
    if "rsi" in enabled_features:
        rsi_period = feature_config.get("rsi_period", 14)
        df.ta.rsi(length=rsi_period, append=True)
        log.debug(f"Calculated RSI({rsi_period})")

    if "macd" in enabled_features:
        macd_fast = feature_config.get("macd_fast_period", 12)
        macd_slow = feature_config.get("macd_slow_period", 26)
        macd_signal = feature_config.get("macd_signal_period", 9)
        df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)
        log.debug(f"Calculated MACD({macd_fast},{macd_slow},{macd_signal})")

    if "bbands" in enabled_features:
        bb_len = feature_config.get("bbands_period", 20)
        bb_std = feature_config.get("bbands_std_dev", 2)
        df.ta.bbands(length=bb_len, std=bb_std, append=True)
        log.debug(f"Calculated BBands({bb_len},{bb_std})")

    if "atr" in enabled_features:
        atr_period = config.get_int(
            "backtest.atr_period", 14
        )  # Reuse from backtest for consistency
        df.ta.atr(length=atr_period, append=True)
        log.debug(f"Calculated ATR({atr_period})")

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
        log.error(f"Configured features not generated: {missing_features}")
        log.error(f"Available columns after generation: {df.columns.tolist()}")
        raise ValueError(f"Missing required features: {missing_features}")

    # Drop rows with NaNs introduced by feature calculation
    initial_rows = len(df)
    df.dropna(inplace=True)
    log.info(f"Dropped {initial_rows - len(df)} rows due to NaNs after feature generation.")
    log.info(f"Feature generation complete. {len(df)} rows remaining.")

    return df


def generate_labels(df: pd.DataFrame, config: "ConfigManager") -> pd.DataFrame:
    """Generate the target variable (label) based on future price movement.

    Args:
        df: Input DataFrame containing OHLCV and feature data
        config: Configuration manager instance containing labeling parameters

    Returns:
        DataFrame with added target labels
    """
    log.info("Generating labels...")
    label_config = config.get("training.labeling", {})
    horizon_minutes = label_config.get("target_horizon_minutes")
    threshold_pct = label_config.get("target_threshold_pct")

    if horizon_minutes is None or threshold_pct is None:
        raise ValueError(
            "Labeling horizon and threshold must be set in config ('training.labeling')"
        )

    log.info(f"Labeling parameters: Horizon={horizon_minutes} mins, Threshold={threshold_pct}%")

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
    log.info(f"Dropped {initial_rows - len(df)} rows due to NaNs after label generation.")

    # Log label distribution
    label_counts = df["target"].value_counts(normalize=True) * 100
    log.info(f"Label distribution:\n{label_counts}")

    if len(label_counts) < 2:
        log.warning(
            "Dataset contains only one class after labeling. "
            "Model training might fail or be meaningless."
        )

    log.info(f"Label generation complete. {len(df)} rows remaining.")
    return df.drop(columns=["future_close", "future_pct_change"])  # Clean up intermediate columns


def split_data(
    df: pd.DataFrame, config: "ConfigManager"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data chronologically into training and testing sets.

    Args:
        df: Input DataFrame containing features and labels
        config: Configuration manager instance containing split parameters

    Returns:
        Tuple containing (X_train, X_test, y_train, y_test)
    """
    log.info("Splitting data...")
    feature_list = config.get_list("training.data.feature_list")
    split_ratio = config.get("training.data.train_split_ratio", 0.8)

    if not feature_list:
        raise ValueError("Feature list ('training.data.feature_list') not defined in config.")

    X = df[feature_list]
    y = df["target"]

    split_index = int(len(df) * split_ratio)
    if split_index == 0 or split_index == len(df):
        raise ValueError(f"Train split ratio {split_ratio} results in empty train or test set.")

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    log.info(f"Data split: Train={len(X_train)} rows, Test={len(X_test)} rows")
    train_dist = y_train.value_counts(normalize=True) * 100
    log.info(f"Training Target Distribution:\n{train_dist}")
    test_dist = y_test.value_counts(normalize=True) * 100
    log.info(f"Testing Target Distribution:\n{test_dist}")

    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: "ConfigManager",
) -> xgb.XGBClassifier:
    """Train an XGBoost model with the provided data.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: Configuration manager instance containing model parameters

    Returns:
        Trained XGBoost classifier model
    """
    log.info("Training XGBoost model...")
    model_config = config.get("training.model", {})
    model_params = model_config.get("params", {})
    model_type = model_config.get("type")

    if model_type != "xgboost":
        raise ValueError(
            f"Model type in config is '{model_type}', but this script trains XGBoost."
        )

    if not model_params:
        log.warning(
            "No model parameters ('training.model.params') found in config. "
            "Using XGBoost defaults."
        )

    # Check target balance and set scale_pos_weight if needed
    balance = y_train.value_counts()
    if 0 in balance and 1 in balance and balance[0] > 0:
        scale_pos_weight = balance[0] / balance[1]
        log.info(f"Target imbalance detected. Setting scale_pos_weight={scale_pos_weight:.2f}")
        model_params["scale_pos_weight"] = scale_pos_weight
    else:
        log.warning("Could not calculate scale_pos_weight (single class or division by zero).")

    model = xgb.XGBClassifier(**model_params, use_label_encoder=False)
    # Pass X_test, y_test to eval_set requires them to be passed into
    # this function. Simplified: fit only on train data for now
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)

    log.info("Model training complete.")
    return model


def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Evaluate model performance on test data.

    Args:
        model: Trained XGBoost classifier
        X_test: Test features
        y_test: Test labels
    """
    log.info("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    log.info("Test Set Performance:")
    log.info("  Accuracy:  {:.4f}".format(accuracy))
    log.info("  Precision: {:.4f}".format(precision))
    log.info("  Recall:    {:.4f}".format(recall))
    log.info("  F1-Score:  {:.4f}".format(f1))
    log.info("  ROC AUC:   {:.4f}".format(roc_auc))


def save_model(model: xgb.XGBClassifier, config: "ConfigManager") -> None:
    """Save the trained model to disk.

    Args:
        model: Trained XGBoost classifier to save
        config: Configuration manager instance containing save path
    """
    save_path = config.get("prediction_service.model_path")
    if not save_path:
        raise ValueError("Model save path ('prediction_service.model_path') not configured.")

    # Ensure the directory exists
    model_dir = os.path.dirname(save_path)
    if model_dir and not os.path.exists(model_dir):
        log.info(f"Creating model directory: {model_dir}")
        os.makedirs(model_dir)

    log.info(f"Saving trained model to: {save_path}")
    try:
        joblib.dump(model, save_path)
        log.info("Model saved successfully.")
    except Exception as e:
        log.exception(f"Error saving model to {save_path}: {e}", exc_info=True)
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
        X_train, X_test, y_train, y_test = split_data(df_labeled, config)

        # 5. Train Model
        # Pass test data for evaluation during training if desired
        model = train_model(X_train, y_train, X_test, y_test, config)  # Pass X_test, y_test

        # 6. Evaluate Model
        evaluate_model(model, X_test, y_test)

        # 7. Save Model
        save_model(model, config)

        log.info("Initial model training script finished successfully.")

    except FileNotFoundError as e:
        log.error(f"Configuration or data file not found: {e}")
    except ValueError as e:
        log.error(f"Configuration error or data issue: {e}")
    except Exception as e:
        log.exception(f"An unexpected error occurred during training: {e}", exc_info=True)


if __name__ == "__main__":
    main()
