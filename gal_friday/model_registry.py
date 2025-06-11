"""Lightweight registry for loading imputation models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib


class ImputationModelRegistry:
    """Registry responsible for loading trained imputation models.

    Parameters
    ----------
    models_dir:
        Directory containing serialized model artifacts. Each model is stored
        as ``<model_key>.pkl`` within this directory.
    """

    def __init__(self, models_dir: str | Path) -> None:
        self.models_dir = Path(models_dir)
        self._cache: Dict[str, Any] = {}

    async def get(self, model_key: str) -> Any:
        """Load a model artifact by key.

        Models are cached after the first load.
        """
        if model_key in self._cache:
            return self._cache[model_key]

        model_path = self.models_dir / f"{model_key}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Imputation model not found: {model_path}")

        model = joblib.load(model_path)
        self._cache[model_key] = model
        return model


def build_ml_features(ohlcv_history: "pd.DataFrame") -> "pd.DataFrame":
    """Convert OHLCV history into features for imputation models.

    This routine computes a small set of technical features suitable for
    learning patterns in the price series.
    """
    import pandas as pd  # Local import to avoid heavy dependency at module import

    df = ohlcv_history.copy()

    df["return_1"] = df["close"].pct_change()
    df["return_5"] = df["close"].pct_change(5)
    df["volatility_5"] = df["close"].rolling(5).std()
    df["volume_sma_5"] = df.get("volume", pd.Series()).rolling(5).mean()
    df["volume_ratio_5"] = df.get("volume") / df["volume_sma_5"]

    return df.dropna().reset_index(drop=True)

