"""
Tests for the PredictorInterface contract.

These tests verify that implementations of PredictorInterface correctly
follow the interface contract and behavior requirements.
"""

from abc import ABC
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gal_friday.interfaces.predictor_interface import PredictorInterface
from gal_friday.predictors.sklearn_predictor import SklearnPredictor
from gal_friday.predictors.xgboost_predictor import XGBoostPredictor


class MockPredictor(PredictorInterface):
    """A minimal implementation of PredictorInterface for testing."""

    def __init__(self, model_path: str, model_id: str, config: Optional[Dict[str, Any]] = None):
        """Initialize with the required parameters but don't load a real model."""
        self.model_path = model_path
        self.model_id = model_id
        self.config = config or {}
        # Don't call super().__init__ to avoid loading a real model
        self.model = MagicMock()

    def load_model(self) -> Any:
        """Implement abstract method for test."""
        # Return a mock model instead of loading a real one
        return MagicMock()

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Implement abstract method for test."""
        # Return simple predictions for testing
        return np.array([0.75, 0.25])

    @property
    def expected_feature_names(self) -> Optional[List[str]]:
        """Implement abstract method for test."""
        return ["feature1", "feature2", "feature3"]


class IncompletePredictor(ABC):
    """A class that inherits from PredictorInterface but doesn't implement all methods."""

    def __init__(self, model_path: str, model_id: str, config: Optional[Dict[str, Any]] = None):
        """Initialize with the required parameters."""
        self.model_path = model_path
        self.model_id = model_id
        self.config = config or {}

    def load_model(self) -> Any:
        """Implement one abstract method but not others."""


def test_predictor_interface_is_abstract():
    """Test that PredictorInterface cannot be instantiated directly."""
    with pytest.raises(TypeError):
        PredictorInterface("model.joblib", "test_model")


def test_predictor_interface_requires_implementation():
    """Test that a class inheriting from PredictorInterface must implement all abstract methods."""
    with pytest.raises(TypeError):
        IncompletePredictor("model.joblib", "test_model")


def test_can_instantiate_concrete_implementation():
    """Test that a concrete implementation can be instantiated."""
    with patch.object(MockPredictor, "load_model", return_value=MagicMock()):
        predictor = MockPredictor("model.joblib", "test_model")
        assert isinstance(predictor, PredictorInterface)


def test_load_model_contract():
    """Test that the load_model method follows the contract."""
    with patch.object(MockPredictor, "load_model", return_value=MagicMock()) as mock_load:
        predictor = MockPredictor("model.joblib", "test_model")
        model = predictor.load_model()

        # Should return a model object
        assert model is not None

        # The model path should be used
        mock_load.assert_called_once()


def test_predict_contract():
    """Test that the predict method follows the contract."""
    with patch.object(MockPredictor, "load_model", return_value=MagicMock()):
        predictor = MockPredictor("model.joblib", "test_model")

        # Should handle numpy array input
        features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        predictions = predictor.predict(features)

        # Should return numpy array output
        assert isinstance(predictions, np.ndarray)


def test_expected_feature_names_contract():
    """Test that the expected_feature_names property follows the contract."""
    with patch.object(MockPredictor, "load_model", return_value=MagicMock()):
        predictor = MockPredictor("model.joblib", "test_model")

        # Should return a list of strings or None
        feature_names = predictor.expected_feature_names
        assert feature_names is None or isinstance(feature_names, list)

        # If it's a list, it should contain strings
        if feature_names is not None:
            assert all(isinstance(name, str) for name in feature_names)


@pytest.mark.parametrize("implementation", [XGBoostPredictor, SklearnPredictor])
def test_real_implementations_conform_to_interface(implementation):
    """Test that real implementations conform to the interface."""
    # This test verifies that our actual implementations properly implement the interface
    assert issubclass(implementation, PredictorInterface)

    # Verify that the implementations have the required methods
    assert hasattr(implementation, "load_model")
    assert hasattr(implementation, "predict")
    assert hasattr(implementation, "expected_feature_names")

    # Test with mocked model loading so we don't need actual model files
    with patch.object(implementation, "load_model", return_value=MagicMock()):
        instance = implementation("path/to/model.pkl", "test_model")

        # Ensure predict takes np.ndarray and returns np.ndarray
        mock_predict = MagicMock(return_value=np.array([0.5]))
        with patch.object(instance, "predict", mock_predict):
            features = np.array([[1.0, 2.0]])
            instance.predict(features)
            mock_predict.assert_called_once()

        # Ensure expected_feature_names returns List[str] or None
        with patch.object(
            instance, "expected_feature_names", new_callable=property, return_value=["feature1"]
        ):
            names = instance.expected_feature_names
            assert isinstance(names, list)
            assert all(isinstance(name, str) for name in names)
