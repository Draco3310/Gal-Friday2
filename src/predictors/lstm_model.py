from src.predictors.base_predictor import BasePredictor
import tensorflow as tf
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LSTMPredictor(BasePredictor):
    def load_model(self, model_path: str):
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"LSTM model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            self.model = None

    def predict(self, enriched_df: pd.DataFrame) -> float:
        if not self.model:
            return 0.0

        feature_names = self.config.get('feature_names')
        timesteps = self.config.get('timesteps', 10)

        if not feature_names:
            logger.warning("LSTM requires 'feature_names' in config.")
            return 0.0

        if len(enriched_df) < timesteps:
            logger.warning(f"Not enough data for LSTM: {len(enriched_df)} < {timesteps}")
            return 0.0

        try:
            data = enriched_df[feature_names].iloc[-timesteps:].values
            X = data.reshape(1, timesteps, len(feature_names))
            pred = self.model.predict(X, verbose=0)
            return float(pred[0][0])
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return 0.0
