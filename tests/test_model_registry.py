"""Tests for Model Registry functionality."""

import asyncio
import shutil
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb
from rich import print as rich_print
from sklearn.preprocessing import StandardScaler

from gal_friday.model_lifecycle import (
    ModelArtifact,
    ModelMetadata,
    ModelRegistry,
    ModelStage,
)


class MockConfig:
    """Mock configuration for testing."""
    def __init__(self):
        self.config = {
            "model_registry.storage_path": "./test_models",
            "model_registry.use_cloud_storage": False,
        }

    def get(self, key, default=None):
        return self.config.get(key, default)

    def get_bool(self, key, default=False):
        return bool(self.config.get(key, default))


class MockLogger:
    """Mock logger for testing."""
    def info(self, msg, **kwargs):
        rich_print(f"INFO: {msg}")

    def warning(self, msg, **kwargs):
        rich_print(f"WARNING: {msg}")

    def error(self, msg, **kwargs):
        rich_print(f"ERROR: {msg}")

    def exception(self, msg, **kwargs):
        rich_print(f"EXCEPTION: {msg}")


class MockModelRepo:
    """Mock model repository for testing."""
    def __init__(self):
        self.models = {}

    async def save_model(self, metadata):
        self.models[metadata.model_id] = metadata
        return metadata.model_id

    async def get_model_by_id(self, model_id):
        return self.models.get(model_id)

    async def get_model_by_version(self, model_name, version):
        for model in self.models.values():
            if model.model_name == model_name and model.version == version:
                return model
        return None

    async def get_model_by_stage(self, model_name, stage):
        for model in self.models.values():
            if model.model_name == model_name and model.stage == stage:
                return model
        return None

    async def get_latest_model(self, model_name):
        models = [m for m in self.models.values() if m.model_name == model_name]
        if models:
            return sorted(models, key=lambda m: m.created_at, reverse=True)[0]
        return None

    async def list_models(self, model_name=None, stage=None):
        results = []
        for model in self.models.values():
            if model_name and model.model_name != model_name:
                continue
            if stage and model.stage != stage:
                continue
            results.append(model)
        return results

    async def update_model_stage(self, model_id, new_stage, updated_by):
        if model_id in self.models:
            self.models[model_id].stage = new_stage
            return True
        return False


class MockSecrets:
    """Mock secrets manager."""


@pytest.mark.asyncio
async def test_model_registry():
    """Test basic model registry functionality."""
    # Setup
    config = MockConfig()
    logger = MockLogger()
    model_repo = MockModelRepo()
    secrets = MockSecrets()

    # Clean up any previous test artifacts
    test_path = Path("./test_models")
    if test_path.exists():
        shutil.rmtree(test_path)

    # Create registry
    registry = ModelRegistry(config, model_repo, logger, secrets)

    # Create a simple XGBoost model
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)

    model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
    model.fit(X_train, y_train)

    # Create preprocessor
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Create model artifact
    artifact = ModelArtifact(
        model=model,
        preprocessor=scaler,
        feature_names=["feature1", "feature2", "feature3", "feature4", "feature5"],
    )

    # Add metadata
    artifact.metadata = ModelMetadata(
        model_type="xgboost",
        metrics={"mae": 0.05, "rmse": 0.08},
        hyperparameters={"n_estimators": 10, "max_depth": 3},
    )

    # Test 1: Register model
    rich_print("Test 1: Registering model...")
    model_id = await registry.register_model(artifact, "price_predictor")
    assert model_id is not None
    rich_print(f"✓ Model registered with ID: {model_id}")

    # Test 2: Get model by name
    rich_print("\nTest 2: Retrieving model...")
    retrieved = await registry.get_model("price_predictor")
    assert retrieved is not None
    assert retrieved.metadata.model_name == "price_predictor"
    assert retrieved.metadata.version == "1.0.0"
    rich_print(f"✓ Model retrieved: {retrieved.metadata.model_name} v{retrieved.metadata.version}")

    # Test 3: Register another version
    rich_print("\nTest 3: Registering new version...")
    artifact2 = ModelArtifact(
        model=model,
        preprocessor=scaler,
        feature_names=["feature1", "feature2", "feature3", "feature4", "feature5"],
    )
    model_id2 = await registry.register_model(artifact2, "price_predictor")
    assert model_id2 is not None

    retrieved2 = await registry.get_model("price_predictor")
    assert retrieved2.metadata.version == "1.0.1"
    rich_print(f"✓ New version registered: v{retrieved2.metadata.version}")

    # Test 4: List models
    rich_print("\nTest 4: Listing models...")
    models = await registry.list_models("price_predictor")
    assert len(models) == 2
    rich_print(f"✓ Found {len(models)} models")

    # Test 5: Promote model
    rich_print("\nTest 5: Promoting model to production...")
    success = await registry.promote_model(model_id2, ModelStage.PRODUCTION)
    assert success

    prod_model = await registry.get_model("price_predictor", stage=ModelStage.PRODUCTION)
    assert prod_model is not None
    assert prod_model.metadata.stage == ModelStage.PRODUCTION
    rich_print(f"✓ Model promoted to {prod_model.metadata.stage.value}")

    # Clean up
    shutil.rmtree(test_path)
    rich_print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_model_registry())
