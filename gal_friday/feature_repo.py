import pandas as pd
from collections import defaultdict
from typing import Dict

_recent_features: Dict[str, pd.DataFrame] = defaultdict(pd.DataFrame)


async def fetch_latest_features(trading_pair: str, limit: int = 200) -> pd.DataFrame | None:
    """Return the most recent computed features for ``trading_pair``."""
    df = _recent_features.get(trading_pair)
    if df is None or df.empty:
        return None
    df = df.sort_index()
    return df.tail(limit)


def store_features(trading_pair: str, features: dict[str, float]) -> None:
    """Append a row of features for ``trading_pair`` to the store."""
    df = _recent_features[trading_pair]
    new_row = pd.DataFrame([features])
    _recent_features[trading_pair] = pd.concat([df, new_row], ignore_index=True)
