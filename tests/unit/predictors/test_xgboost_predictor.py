"""
Tests for the XGBoostPredictor implementation.

These tests verify that the XGBoostPredictor correctly implements the
PredictorInterface for XGBoost models.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xgboost as xgb

from gal_friday.predictors.xgboost_predictor import XGBoostPredictor


@pytest.fixture
def mock_xgboost_model():
    """Create a mock XGBoost model."""
    mock_model = MagicMock(spec=xgb.Booster)
    mock_model.predict.return_value = np.array([0.7, 0.3])
    return mock_model


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def temp_model_file():
    """Create a temporary file for model saving/loading tests."""
    # Create a simple XGBoost model
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)
    dtrain = xgb.DMatrix(X, label=y)
    params = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}
    model = xgb.train(params, dtrain, num_boost_round=2)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".xgb", delete=False) as temp:
        model.save_model(temp.name)
        temp_path = temp.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_xgboost_predictor_initialization():
    """Test that XGBoostPredictor initializes correctly."""
    with patch("xgboost.Booster", return_value=MagicMock()):
        with patch.object(XGBoostPredictor, "load_model", return_value=MagicMock()):
            predictor = XGBoostPredictor(
                model_path="path/to/model.xgb",
                model_id="xgb_test_model",
                config={"feature_names": ["f1", "f2", "f3"]},
            )

            assert predictor.model_path == "path/to/model.xgb"
            assert predictor.model_id == "xgb_test_model"
            assert predictor.config.get("feature_names") == ["f1", "f2", "f3"]


def test_load_model_with_real_file(temp_model_file):
    """Test loading an actual XGBoost model from a file."""
    predictor = XGBoostPredictor(model_path=temp_model_file, model_id="test_real_model")

    assert predictor.model is not None
    assert isinstance(predictor.model, xgb.Booster)


def test_load_model_file_not_found():
    """Test error handling when model file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        XGBoostPredictor(model_path="non_existent_file.xgb", model_id="test_error_model")


def test_predict_method(mock_xgboost_model, sample_features):
    """Test the predict method returns correct format."""
    with patch.object(XGBoostPredictor, "load_model", return_value=mock_xgboost_model):
        predictor = XGBoostPredictor(model_path="path/to/model.xgb", model_id="test_predict_model")

        # Test prediction
        result = predictor.predict(sample_features)

        # Verify model was called with correct parameters
        mock_xgboost_model.predict.assert_called_once()

        # Verify result format
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)  # Based on our mock setup


def test_expected_feature_names():
    """Test the expected_feature_names property."""
    # Case 1: Feature names provided in config using model_feature_names key
    with patch("os.path.exists", return_value=True):
        with patch("xgboost.Booster"):
            predictor = XGBoostPredictor(
                model_path="path/to/model.xgb",
                model_id="test_features_model",
                config={"model_feature_names": ["f1", "f2", "f3"]},
            )

            # Manually set the expected features (bypassing the mocked load_model)
            predictor._expected_features = ["f1", "f2", "f3"]

            assert predictor.expected_feature_names == ["f1", "f2", "f3"]

    # Case 2: Feature names using feature_names key (alternate config key)
    with patch("os.path.exists", return_value=True):
        with patch("xgboost.Booster"):
            predictor = XGBoostPredictor(
                model_path="path/to/model.xgb",
                model_id="test_features_model2",
                config={"feature_names": ["alt_f1", "alt_f2", "alt_f3"]},
            )

            # Manually set the expected features
            predictor._expected_features = ["alt_f1", "alt_f2", "alt_f3"]

            assert predictor.expected_feature_names == ["alt_f1", "alt_f2", "alt_f3"]

    # Case 3: No feature names available
    with patch("os.path.exists", return_value=True):
        with patch("xgboost.Booster"):
            predictor = XGBoostPredictor(
                model_path="path/to/model.xgb", model_id="test_features_model3"
            )

            # Set empty feature list
            predictor._expected_features = []

            assert predictor.expected_feature_names is None


def test_real_prediction_with_real_model(temp_model_file):
    """Test making a real prediction with a real model."""
    predictor = XGBoostPredictor(model_path=temp_model_file, model_id="test_real_predict")

    # Create test features
    features = np.random.rand(5, 3)

    # Make prediction
    result = predictor.predict(features)

    # Verify result
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert np.all((result >= 0) & (result <= 1))  # Binary logistic outputs should be probabilities
