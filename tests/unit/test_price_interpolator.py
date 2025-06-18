from datetime import UTC, timedelta
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

# Provide a dummy logger to satisfy dependencies
stub_logger = types.ModuleType("gal_friday.logger_service")


class _DummyLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass


stub_logger.LoggerService = _DummyLogger
sys.modules.setdefault("gal_friday.logger_service", stub_logger)

stub_providers = types.ModuleType("gal_friday.providers")


class _DummyProvider:
    pass


stub_providers.APIDataProvider = _DummyProvider
stub_providers.DatabaseDataProvider = _DummyProvider
stub_providers.LocalFileDataProvider = _DummyProvider
sys.modules.setdefault("gal_friday.providers", stub_providers)

SPEC_PATH = (
    Path(__file__).resolve().parents[2]
    / "gal_friday"
    / "simulated_market_price_service.py"
)
spec = importlib.util.spec_from_file_location(
    "gal_friday.simulated_market_price_service",
    SPEC_PATH,
)
price_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(price_module)

PriceInterpolator = price_module.PriceInterpolator
DataGap = price_module.DataGap


def _sample_data(start: str, periods: int, slope: float = 1.0) -> pd.DataFrame:
    times = pd.date_range(start, periods=periods, freq="1min", tz=UTC)
    base = np.arange(periods, dtype=float) * slope + 100
    return pd.DataFrame(
        {
            "timestamp": times,
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base,
            "volume": np.arange(periods, dtype=float) * 10 + 100,
        },
    )


@pytest.mark.asyncio
async def test_spline_interpolation_differs_from_linear(mock_logger) -> None:
    interpolator = PriceInterpolator({})

    pre = _sample_data("2023-01-01 00:00", 5, slope=1.0)
    post = _sample_data("2023-01-01 00:07", 5, slope=2.0)
    df = pd.concat([pre, post], ignore_index=True)

    gap = DataGap(
        start_time=pre.iloc[-1]["timestamp"],
        end_time=post.iloc[0]["timestamp"],
        gap_type="data_gap",
        duration_minutes=3,
        before_price=pre.iloc[-1]["close"],
        after_price=post.iloc[0]["open"],
    )

    before_idx = df[pd.to_datetime(df["timestamp"]) <= gap.start_time].index[-1]
    after_idx = df[pd.to_datetime(df["timestamp"]) >= gap.end_time].index[0]
    timestamps = [gap.start_time + timedelta(minutes=1), gap.end_time - timedelta(minutes=1)]

    linear = await interpolator._linear_interpolation(df, before_idx, after_idx, timestamps, gap)
    spline = await interpolator._spline_interpolation(df, before_idx, after_idx, timestamps, gap)

    linear_close = [p["close"] for p in linear]
    spline_close = [p["close"] for p in spline]

    assert len(linear_close) == len(spline_close)
    assert spline[0]["interpolated"] is True
    assert not np.allclose(linear_close, spline_close)
