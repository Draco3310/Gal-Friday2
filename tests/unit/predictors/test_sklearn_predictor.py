"""
Tests for the SklearnPredictor implementation.

These tests verify that the SklearnPredictor correctly implements the
PredictorInterface for scikit-learn models.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from gal_friday.predictors.sklearn_predictor import SklearnPredictor


@pytest.fixture
def mock_sklearn_model():
    """Create a mock scikit-learn classifier model."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4]])
    mock_model.predict.return_value = np.array([1, 0])
    return mock_model


@pytest.fixture
def mock_sklearn_regressor():
    """Create a mock scikit-learn regressor model."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.7, 0.3])
    # Regressors typically don't have predict_proba
    return mock_model


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def temp_model_file():
    """Create a temporary file with a real sklearn model for testing."""
    # Create a simple logistic regression model
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)
    model = LogisticRegression().fit(X, y)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp:
        joblib.dump(model, temp.name)
        temp_path = temp.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_sklearn_predictor_initialization():
    """Test that SklearnPredictor initializes correctly."""
    with patch("joblib.load", return_value=MagicMock()):
        with patch.object(SklearnPredictor, "load_model", return_value=MagicMock()):
            predictor = SklearnPredictor(
                model_path="path/to/model.joblib",
                model_id="sklearn_test_model",
                config={"model_feature_names": ["f1", "f2", "f3"]},
            )

            assert predictor.model_path == "path/to/model.joblib"
            assert predictor.model_id == "sklearn_test_model"
            assert predictor.config.get("model_feature_names") == ["f1", "f2", "f3"]


def test_load_model_with_real_file(temp_model_file):
    """Test loading an actual scikit-learn model from a file."""
    predictor = SklearnPredictor(model_path=temp_model_file, model_id="test_real_model")

    assert predictor.model is not None
    assert isinstance(predictor.model, LogisticRegression)


def test_load_model_file_not_found():
    """Test error handling when model file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        SklearnPredictor(model_path="non_existent_file.joblib", model_id="test_error_model")


def test_predict_classifier(mock_sklearn_model, sample_features):
    """Test the predict method with a classifier model (using predict_proba)."""
    with patch.object(SklearnPredictor, "load_model", return_value=mock_sklearn_model):
        predictor = SklearnPredictor(
            model_path="path/to/model.joblib", model_id="test_predict_classifier"
        )

        # Test prediction
        result = predictor.predict(sample_features)

        # Verify model was called with correct method
        mock_sklearn_model.predict_proba.assert_called_once_with(sample_features)

        # Verify result format
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)  # 2 samples, 2 classes


def test_predict_regressor(mock_sklearn_regressor, sample_features):
    """Test the predict method with a regressor model."""
    with patch.object(SklearnPredictor, "load_model", return_value=mock_sklearn_regressor):
        predictor = SklearnPredictor(
            model_path="path/to/model.joblib", model_id="test_predict_regressor"
        )

        # Ensure mock_sklearn_regressor does not have predict_proba
        if hasattr(mock_sklearn_regressor, "predict_proba"):
            delattr(mock_sklearn_regressor, "predict_proba")

        # Test prediction
        result = predictor.predict(sample_features)

        # Verify model was called with correct method
        mock_sklearn_regressor.predict.assert_called_once_with(sample_features)

        # Verify result format
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 1)  # 2 samples, 1 output dimension


def test_predict_invalid_model():
    """Test prediction with an invalid model (no predict methods)."""
    # Create an object without predict or predict_proba methods
    class InvalidModel:
        pass

    mock_model = InvalidModel()

    with patch.object(SklearnPredictor, "load_model", return_value=mock_model):
        predictor = SklearnPredictor(
            model_path="path/to/model.joblib", model_id="test_invalid_model"
        )

        # Prediction should raise TypeError
        with pytest.raises(TypeError, match="has no predict or predict_proba method"):
            predictor.predict(np.array([[1, 2, 3]]))


def test_expected_feature_names_from_model():
    """Test expected_feature_names from model attributes."""
    # Create a mock model with feature_names_in_ attribute
    mock_model = MagicMock()
    mock_model.feature_names_in_ = ["feature1", "feature2", "feature3"]

    # Fix: Need to create a complete mock that includes setting _expected_features
    with patch("joblib.load", return_value=mock_model):
        # Don't mock load_model so it actually runs and sets _expected_features
        predictor = SklearnPredictor(
            model_path="path/to/model.joblib", model_id="test_features_model"
        )

        # Check that feature names were read from model
        assert predictor.expected_feature_names == ["feature1", "feature2", "feature3"]


def test_expected_feature_names_from_config():
    """Test expected_feature_names from config when model doesn't have them."""

    class ModelWithoutFeatureNames:
        def predict(self, X):
            return np.array([0.1, 0.2])

    model_without_features = ModelWithoutFeatureNames()
    assert not hasattr(model_without_features, "feature_names_in_")

    with patch("joblib.load", return_value=model_without_features):
        predictor = SklearnPredictor(
            model_path="path/to/model.joblib",
            model_id="test_features_config",
            config={"model_feature_names": ["config_f1", "config_f2"]},
        )
        assert predictor.expected_feature_names == ["config_f1", "config_f2"]


def test_real_prediction_with_real_model(temp_model_file, sample_features):
    """Test making a real prediction with a real scikit-learn model."""
    predictor = SklearnPredictor(model_path=temp_model_file, model_id="test_real_predict")

    # Make prediction
    result = predictor.predict(sample_features)

    # Verify result
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)  # 2 samples, 2 classes (binary classifier probabilities)
    assert np.all((result >= 0) & (result <= 1))  # Probabilities should be between 0 and 1
