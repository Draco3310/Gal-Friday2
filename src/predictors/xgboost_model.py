from src.predictors.base_predictor import BasePredictor
import xgboost as xgb
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class XGBoostPredictor(BasePredictor):
    def load_model(self, model_path: str):
        try:
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            logger.info(f"XGBoost model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            self.model = None

    def predict(self, enriched_df: pd.DataFrame) -> float:
        if not self.model:
            return 0.0

        feature_names = self.config.get('feature_names')

        if feature_names:
            available_cols = set(enriched_df.columns)
            missing = [f for f in feature_names if f not in available_cols]
            if missing:
                logger.warning(f"Missing features: {missing}")
                return 0.0
            X = enriched_df[feature_names].iloc[[-1]].values
        else:
            # Try to use all numeric columns
            X = enriched_df.select_dtypes(include=[np.number]).iloc[[-1]].values
            feature_names = list(enriched_df.select_dtypes(include=[np.number]).columns)

        try:
            # DMatrix is often required for Booster
            dmatrix = xgb.DMatrix(X, feature_names=feature_names)
            preds = self.model.predict(dmatrix)
            # Binary classification output
            return float(preds[0])
        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")
            return 0.0
