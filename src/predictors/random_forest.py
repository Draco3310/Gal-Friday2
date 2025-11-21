from src.predictors.base_predictor import BasePredictor
import joblib
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RandomForestPredictor(BasePredictor):
    def load_model(self, model_path: str):
        self.model = joblib.load(model_path)
        logger.info(f"RandomForest model loaded from {model_path}")

    def predict(self, enriched_df: pd.DataFrame) -> float:
        if not self.model:
            logger.warning("Model not loaded, returning 0.0 confidence.")
            return 0.0

        # Get feature names from config or model attribute if available
        feature_names = self.config.get('feature_names')
        if not feature_names and hasattr(self.model, 'feature_names_in_'):
            feature_names = list(self.model.feature_names_in_)

        if feature_names:
            # Validate columns
            available_cols = set(enriched_df.columns)
            missing = [f for f in feature_names if f not in available_cols]
            if missing:
                logger.warning(f"Missing features for prediction: {missing}")
                return 0.0 # Cannot predict

            # Extract last row features in correct order
            X = enriched_df[feature_names].iloc[[-1]].values
        else:
            # Fallback: use all numeric columns (risky but functional for simple setups)
            X = enriched_df.select_dtypes(include=[np.number]).iloc[[-1]].values

        try:
            # predict_proba returns [prob_class_0, prob_class_1]
            probs = self.model.predict_proba(X)
            # Assuming binary classification: 0=Hold/Sell, 1=Buy
            # If multi-class, we might need specific logic
            if probs.shape[1] >= 2:
                return float(probs[0][1])
            else:
                return float(probs[0][0])
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0
