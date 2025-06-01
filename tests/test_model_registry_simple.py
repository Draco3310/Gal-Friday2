"""Simple test to demonstrate Model Registry structure."""

import json
import pickle
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from rich import print as rich_print


# Simplified versions of the classes for demonstration
class ModelStage:
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

class ModelStatus:
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"

class ModelMetadata:
    def __init__(self):
        self.model_id = str(uuid.uuid4())
        self.model_name = ""
        self.version = ""
        self.model_type = ""
        self.created_at = datetime.now(UTC)
        self.stage = ModelStage.DEVELOPMENT
        self.status = ModelStatus.READY
        self.metrics = {}
        self.hyperparameters = {}
        self.feature_importance = {}
        self.artifact_path = None
        self.features = []

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "model_type": self.model_type,
            "created_at": self.created_at.isoformat(),
            "stage": self.stage,
            "status": self.status,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "feature_importance": self.feature_importance,
            "artifact_path": self.artifact_path,
            "features": self.features,
        }

class SimpleModel:
    """A simple mock model for testing."""
    def __init__(self, name):
        self.name = name
        self.params = {"learning_rate": 0.1, "n_estimators": 100}

    def predict(self, X):
        return [0.5] * len(X)

def demonstrate_model_registry():
    """Demonstrate the Model Registry structure and workflow."""
    rich_print("=== Gal-Friday Model Registry Demonstration ===\n")

    # 1. Create storage directory
    storage_path = Path("./demo_models")
    if storage_path.exists():
        shutil.rmtree(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)

    # 2. Create a model and metadata
    model = SimpleModel("price_predictor")

    metadata = ModelMetadata()
    metadata.model_name = "price_predictor"
    metadata.version = "1.0.0"
    metadata.model_type = "xgboost"
    metadata.metrics = {
        "train_mae": 0.045,
        "train_rmse": 0.067,
        "val_mae": 0.052,
        "val_rmse": 0.074,
    }
    metadata.hyperparameters = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
    }
    metadata.features = ["price_lag_1", "price_lag_2", "volume", "rsi", "macd"]

    # 3. Save model artifact
    artifact_path = storage_path / metadata.model_name / metadata.version
    artifact_path.mkdir(parents=True, exist_ok=True)

    # Save model
    with open(artifact_path / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save metadata
    with open(artifact_path / "metadata.json", "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    # Save features
    with open(artifact_path / "features.json", "w") as f:
        json.dump(metadata.features, f)

    metadata.artifact_path = str(artifact_path)

    rich_print("✓ Model artifact saved to:", artifact_path)
    rich_print(f"  - Model: {metadata.model_name} v{metadata.version}")
    rich_print(f"  - Type: {metadata.model_type}")
    rich_print(f"  - Features: {len(metadata.features)}")
    rich_print(f"  - Metrics: {metadata.metrics}")

    # 4. Demonstrate model promotion workflow
    rich_print("\n=== Model Lifecycle Stages ===")
    rich_print("1. Development: Model is being trained and evaluated")
    rich_print("2. Staging: Model passed tests and ready for A/B testing")
    rich_print("3. Production: Model is serving live predictions")
    rich_print("4. Archived: Model is retired but kept for reference")

    rich_print(f"\nCurrent stage: {metadata.stage}")

    # Simulate promotion
    rich_print("\n✓ Promoting model to STAGING...")
    metadata.stage = ModelStage.STAGING

    rich_print("✓ Running A/B tests...")
    rich_print("  - Control: price_predictor v0.9.0")
    rich_print("  - Treatment: price_predictor v1.0.0")
    rich_print("  - Result: +5% improvement in accuracy")

    rich_print("\n✓ Promoting model to PRODUCTION...")
    metadata.stage = ModelStage.PRODUCTION
    metadata.status = ModelStatus.DEPLOYED

    # 5. Load model back
    rich_print("\n=== Loading Model from Registry ===")

    # Load model
    with open(artifact_path / "model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    # Load metadata
    with open(artifact_path / "metadata.json") as f:
        loaded_metadata = json.load(f)

    rich_print(f"✓ Loaded model: {loaded_metadata['model_name']} v{loaded_metadata['version']}")
    rich_print(f"  - Stage: {loaded_metadata['stage']}")
    rich_print(f"  - Status: {loaded_metadata['status']}")

    # 6. Make prediction
    rich_print("\n=== Making Predictions ===")
    test_data = [[0.5, 0.6, 1000, 45, 0.02]]  # Mock feature values
    prediction = loaded_model.predict(test_data)
    rich_print(f"✓ Prediction for test data: {prediction[0]}")

    # Clean up
    shutil.rmtree(storage_path)

    rich_print("\n✅ Model Registry demonstration complete!")

    # Show benefits
    rich_print("\n=== Benefits of Model Registry ===")
    rich_print("1. Version Control: Track all model versions with full history")
    rich_print("2. Metadata Tracking: Store metrics, parameters, and features")
    rich_print("3. Lifecycle Management: Promote models through stages safely")
    rich_print("4. A/B Testing: Compare models before production deployment")
    rich_print("5. Rollback Capability: Quickly revert to previous versions")
    rich_print("6. Audit Trail: Complete history of all model changes")


if __name__ == "__main__":
    demonstrate_model_registry()
