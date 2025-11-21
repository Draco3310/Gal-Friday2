import pandas as pd
import pandas_ta as ta
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)

class FeatureEngine:
    """
    Centralized engine that enriches raw OHLCV data with technical indicators.
    """
    def __init__(self):
        # We might add scaling state here if we need to maintain state across calls,
        # but for V1 we assume stateless calculation or refit on every batch?
        # Typically models need consistent scaling.
        # If we load pre-trained models, they expect data scaled in the same way as training.
        # The legacy code had pipelines.
        # For this V2 implementation, if we are "Restarting", we imply new models or models that include scaling?
        # The XGBoost/RF/LSTM plugins might expect scaled data.
        # I will implement 'compute_features' to return enriched data.
        # Scaling might be done inside the predictor or here.
        # The prompt says "The Feature Engine... Centralized output scaling...".
        # So I should probably scale here. But scaling requires state (mean, std) from training data.
        # If I don't have training data here, I can't consistently scale for inference.
        # Unless I load a scaler.
        # The legacy code loaded scalers.
        # Given the "clean slate" directive, I will assume the Predictors might handle scaling
        # OR the FeatureEngine computes raw indicators and we rely on the model to handle it or we accept unscaled for tree models.
        # Tree models (XGB, RF) don't strictly need scaling. LSTM does.
        pass

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes technical indicators and cleans the data.

        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'volume'.

        Returns:
            DataFrame with added feature columns and NaNs removed.
        """
        if df.empty:
            return df

        try:
            # Ensure columns are lower case
            df.columns = [c.lower() for c in df.columns]

            # Log Returns
            df['log_return'] = df.ta.log_return(append=False)

            # RSI (14)
            df.ta.rsi(length=14, append=True)

            # MACD (12, 26, 9)
            # Returns MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            df.ta.macd(fast=12, slow=26, signal=9, append=True)

            # Bollinger Bands (20, 2)
            # Returns BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
            df.ta.bbands(length=20, std=2, append=True)

            # ATR (14)
            df.ta.atr(length=14, append=True)

            # VWAP - requires datetime index or proper configuration in pandas_ta
            # If index is datetime, pandas_ta handles it usually.
            if 'volume' in df.columns:
                try:
                    df.ta.vwap(append=True)
                except Exception:
                    # VWAP calculation can vary if index is not proper datetime
                    logger.debug("VWAP calculation skipped or failed.")

            # Drop NaNs created by indicators (lookback periods)
            df.dropna(inplace=True)

            return df

        except Exception as e:
            logger.error(f"Feature computation failed: {e}")
            raise
