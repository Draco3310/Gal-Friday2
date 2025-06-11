import importlib.util
import sys
import types
from datetime import UTC, timedelta
from pathlib import Path
import pytest

import numpy as np
import pandas as pd

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

SPEC_PATH = Path(__file__).resolve().parents[2] / "gal_friday" / "data_ingestion" / "gap_detector.py"
spec = importlib.util.spec_from_file_location("gap_detector", SPEC_PATH)
gap_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gap_module)

DataInterpolator = gap_module.DataInterpolator
InterpolationConfig = gap_module.InterpolationConfig
GapInfo = gap_module.GapInfo


def _create_interpolator(logger):
    config = InterpolationConfig()
    return DataInterpolator(config, logger)


def _sample_data(start, periods, slope=1.0, vol=0.5):
    """Create simple OHLCV test data with configurable volatility."""
    times = pd.date_range(start, periods=periods, freq="1min", tz=UTC)
    base = np.arange(periods, dtype=float) * slope + 100
    data = pd.DataFrame(
        {
            "open": base,
            "high": base + vol,
            "low": base - vol,
            "close": base,
        },
        index=times,
    )
    return data


def test_spline_interpolation_non_linear_small_gap(mock_logger):
    interpolator = _create_interpolator(mock_logger)

    pre = _sample_data("2023-01-01 00:00", 5, slope=1.0)
    post = _sample_data("2023-01-01 00:07", 5, slope=2.0)
    combined = pd.concat([pre, post])

    time_index = pd.date_range("2023-01-01 00:05", periods=2, freq="1min", tz=UTC)

    linear_res = interpolator._linear_interpolation(combined, time_index)
    spline_res = interpolator._spline_interpolation(combined, time_index)

    assert list(spline_res.columns) == list(linear_res.columns)
    assert not np.allclose(spline_res["close"], linear_res["close"])


def test_volatility_adjusted_interpolation_non_linear_large_gap(mock_logger):
    interpolator = _create_interpolator(mock_logger)

    pre = _sample_data("2023-01-01 00:00", 5, slope=1.0)
    post = _sample_data("2023-01-01 00:10", 5, slope=2.0)
    combined = pd.concat([pre, post])

    time_index = pd.date_range("2023-01-01 00:05", periods=5, freq="1min", tz=UTC)

    gap_info = GapInfo(
        start_time=time_index[0] - timedelta(minutes=1),
        end_time=time_index[-1] + timedelta(minutes=1),
        duration=timedelta(minutes=6),
        symbol="XRP/USD",
        data_type="ohlcv",
        gap_size=len(time_index),
        preceding_data_quality=1.0,
        following_data_quality=1.0,
    )

    linear_res = interpolator._linear_interpolation(combined, time_index)
    vol_res = interpolator._volatility_adjusted_interpolation(combined, time_index, gap_info)

    assert list(vol_res.columns) == list(linear_res.columns)
    assert not np.allclose(vol_res["close"], linear_res["close"])
    assert vol_res["close"].diff().diff().abs().sum() > 0


def test_volatility_adjusted_differs_from_spline(mock_logger):
    """Ensure volatility adjustment changes the spline result."""
    pytest.importorskip("pandas_ta")
    interpolator = _create_interpolator(mock_logger)

    pre = _sample_data("2023-01-01 00:00", 5, slope=1.0, vol=0.5)
    post = _sample_data("2023-01-01 00:10", 5, slope=2.0, vol=0.5)
    combined = pd.concat([pre, post])

    time_index = pd.date_range("2023-01-01 00:05", periods=5, freq="1min", tz=UTC)

    gap_info = GapInfo(
        start_time=time_index[0] - timedelta(minutes=1),
        end_time=time_index[-1] + timedelta(minutes=1),
        duration=timedelta(minutes=6),
        symbol="XRP/USD",
        data_type="ohlcv",
        gap_size=len(time_index),
        preceding_data_quality=1.0,
        following_data_quality=1.0,
    )

    spline_res = interpolator._spline_interpolation(combined, time_index)
    vol_res = interpolator._volatility_adjusted_interpolation(combined, time_index, gap_info)

    assert not np.allclose(vol_res["close"], spline_res["close"])


def test_volatility_adjusted_scales_with_volatility(mock_logger):
    """Higher volatility should lead to larger adjustments."""
    pytest.importorskip("pandas_ta")
    interpolator = _create_interpolator(mock_logger)

    # Low volatility dataset
    pre_low = _sample_data("2023-01-01 00:00", 5, slope=1.0, vol=0.5)
    post_low = _sample_data("2023-01-01 00:10", 5, slope=2.0, vol=0.5)
    combined_low = pd.concat([pre_low, post_low])
    time_low = pd.date_range("2023-01-01 00:05", periods=5, freq="1min", tz=UTC)

    gap_low = GapInfo(
        start_time=time_low[0] - timedelta(minutes=1),
        end_time=time_low[-1] + timedelta(minutes=1),
        duration=timedelta(minutes=6),
        symbol="XRP/USD",
        data_type="ohlcv",
        gap_size=len(time_low),
        preceding_data_quality=1.0,
        following_data_quality=1.0,
    )

    spline_low = interpolator._spline_interpolation(combined_low, time_low)
    vol_low = interpolator._volatility_adjusted_interpolation(combined_low, time_low, gap_low)
    dev_low = np.abs(vol_low["close"] - spline_low["close"]).mean()

    # High volatility dataset (wider high/low range)
    pre_high = _sample_data("2023-01-02 00:00", 5, slope=1.0, vol=2.0)
    post_high = _sample_data("2023-01-02 00:10", 5, slope=2.0, vol=2.0)
    combined_high = pd.concat([pre_high, post_high])
    time_high = pd.date_range("2023-01-02 00:05", periods=5, freq="1min", tz=UTC)

    gap_high = GapInfo(
        start_time=time_high[0] - timedelta(minutes=1),
        end_time=time_high[-1] + timedelta(minutes=1),
        duration=timedelta(minutes=6),
        symbol="XRP/USD",
        data_type="ohlcv",
        gap_size=len(time_high),
        preceding_data_quality=1.0,
        following_data_quality=1.0,
    )

    spline_high = interpolator._spline_interpolation(combined_high, time_high)
    vol_high = interpolator._volatility_adjusted_interpolation(combined_high, time_high, gap_high)
    dev_high = np.abs(vol_high["close"] - spline_high["close"]).mean()

    assert dev_high > dev_low
