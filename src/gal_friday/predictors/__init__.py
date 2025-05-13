"""
Predictors module for Gal-Friday.

This module contains implementations of different machine learning model predictors.
"""

from ..interfaces.predictor_interface import PredictorInterface
from .sklearn_predictor import SklearnPredictor
from .xgboost_predictor import XGBoostPredictor

__all__ = [
    "PredictorInterface",
    "XGBoostPredictor",
    "SklearnPredictor",
]
