"""Unit tests for _pipeline_compute static methods in FeatureEngine."""

from collections import deque
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pandas_ta as ta  # For comparing results
import pytest

from gal_friday.feature_engine import FeatureEngine


# --- Existing Fixtures ---
@pytest.fixture
def sample_close_prices_long() -> pd.Series:
    """Fixture for a longer sample close price series."""
    return pd.Series([
        10.0, 10.2, 10.1, 10.3, 10.5, 10.4, 10.6, 10.8, 11.0, 10.7,
        10.9, 11.2, 11.5, 11.3, 11.6, 11.8, 11.9, 12.1, 12.3, 12.0,
        12.2, 12.5, 12.8, 12.6, 12.9, 13.1, 13.0, 13.3, 13.5, 13.2,
    ], dtype="float64", name="close") # Added name for clarity

@pytest.fixture
def sample_close_prices_short() -> pd.Series:
    """Fixture for a short sample close price series (less than typical periods)."""
    return pd.Series([10.0, 10.2, 10.1, 10.3, 10.5], dtype="float64", name="close")

@pytest.fixture
def sample_close_prices_with_nan() -> pd.Series:
    """Fixture for a sample close price series with NaNs."""
    return pd.Series([
        10.0, 10.2, np.nan, 10.3, 10.5, 10.4, np.nan, 10.8, 11.0, 10.7,
        10.9, 11.2, 11.5, 11.3, 11.6, 11.8, 11.9, 12.1, 12.3, 12.0,
    ], dtype="float64", name="close")

# --- New Fixtures ---
@pytest.fixture
def sample_ohlcv_data_long() -> pd.DataFrame:
    data_len = 35 # Enough for typical periods like 14, 20, 26
    data = {
        "open": np.array([100 + i + np.sin(i/5) for i in range(data_len)], dtype="float64"),
        "high": np.array([102 + i + np.sin(i/5) + abs(np.random.randn()) for i in range(data_len)], dtype="float64"),
        "low": np.array([98 + i + np.sin(i/5) - abs(np.random.randn()) for i in range(data_len)], dtype="float64"),
        "close": np.array([100 + i + np.cos(i/5) for i in range(data_len)], dtype="float64"), # Use 'close' from sample_close_prices_long for some tests
        "volume": np.array([1000 + i*10 + abs(np.random.randn()*100) for i in range(data_len)], dtype="float64"),
    }
    df = pd.DataFrame(data)
    # Ensure high is max and low is min
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df

@pytest.fixture
def sample_ohlcv_data_short(sample_ohlcv_data_long) -> pd.DataFrame:
    return sample_ohlcv_data_long.head(5)

@pytest.fixture
def sample_ohlcv_data_with_nan(sample_ohlcv_data_long) -> pd.DataFrame:
    df = sample_ohlcv_data_long.copy()
    df.loc[2, "high"] = np.nan
    df.loc[3, "low"] = np.nan
    df.loc[4, "close"] = np.nan
    df.loc[5, "volume"] = np.nan
    return df

@pytest.fixture
def sample_l2_book_valid() -> dict:
    return {
        "bids": [[Decimal("100.00"), Decimal("1.5")], [Decimal("99.95"), Decimal("2.0")]],
        "asks": [[Decimal("100.05"), Decimal("1.2")], [Decimal("100.10"), Decimal("1.8")]],
        "timestamp": datetime.now(UTC),
    }

@pytest.fixture
def sample_l2_book_empty() -> dict:
    return {"bids": [], "asks": [], "timestamp": datetime.now(UTC)}

@pytest.fixture
def sample_l2_book_empty_bids() -> dict:
    return {
        "bids": [],
        "asks": [[Decimal("100.05"), Decimal("1.2")]],
        "timestamp": datetime.now(UTC),
    }

@pytest.fixture
def sample_l2_book_empty_asks() -> dict:
    return {
        "bids": [[Decimal("100.00"), Decimal("1.5")]],
        "asks": [],
        "timestamp": datetime.now(UTC),
    }


@pytest.fixture
def sample_l2_books_series(sample_l2_book_valid, sample_l2_book_empty) -> pd.Series:
    # Timestamps for indexing the series
    idx = pd.to_datetime([
        datetime.now(UTC) - pd.Timedelta(seconds=2),
        datetime.now(UTC) - pd.Timedelta(seconds=1),
        datetime.now(UTC),
    ])
    return pd.Series([sample_l2_book_valid, None, sample_l2_book_empty], index=idx)

@pytest.fixture
def sample_trades_deque() -> deque:
    return deque([
        {"timestamp": datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC), "price": Decimal("100.0"), "volume": Decimal("1.0"), "side": "buy"},
        {"timestamp": datetime(2023, 1, 1, 12, 0, 10, tzinfo=UTC), "price": Decimal("100.1"), "volume": Decimal("0.5"), "side": "sell"},
        {"timestamp": datetime(2023, 1, 1, 12, 0, 20, tzinfo=UTC), "price": Decimal("100.2"), "volume": Decimal("1.2"), "side": "buy"},
        {"timestamp": datetime(2023, 1, 1, 12, 1, 0, tzinfo=UTC), "price": Decimal("100.3"), "volume": Decimal("0.8"), "side": "buy"}, # Next bar
        {"timestamp": datetime(2023, 1, 1, 12, 1, 15, tzinfo=UTC), "price": Decimal("100.2"), "volume": Decimal("0.3"), "side": "sell"},
    ])

@pytest.fixture
def sample_bar_start_times() -> pd.Series:
    return pd.Series([
        datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC),
        datetime(2023, 1, 1, 12, 1, 0, tzinfo=UTC),
        datetime(2023, 1, 1, 12, 2, 0, tzinfo=UTC), # Bar with no trades
    ])


# --- Test Classes ---

# Tests for _pipeline_compute_rsi (existing)
class TestPipelineComputeRSI:
    def test_rsi_basic_calculation(self, sample_close_prices_long):
        period = 14
        expected_rsi = sample_close_prices_long.ta.rsi(length=period)
        expected_rsi.name = f"rsi_{period}"

        actual_rsi = FeatureEngine._pipeline_compute_rsi(sample_close_prices_long, period=period)

        pd.testing.assert_series_equal(actual_rsi, expected_rsi, check_dtype=True)
        assert actual_rsi.name == f"rsi_{period}"

    def test_rsi_insufficient_data(self, sample_close_prices_short):
        period = 14
        expected_rsi = sample_close_prices_short.ta.rsi(length=period)
        expected_rsi.name = f"rsi_{period}"

        actual_rsi = FeatureEngine._pipeline_compute_rsi(sample_close_prices_short, period=period)

        pd.testing.assert_series_equal(actual_rsi, expected_rsi, check_dtype=True)
        assert actual_rsi.name == f"rsi_{period}"
        assert actual_rsi.isna().all()

    def test_rsi_with_nan_input(self, sample_close_prices_with_nan):
        period = 14
        expected_rsi = sample_close_prices_with_nan.ta.rsi(length=period)
        expected_rsi.name = f"rsi_{period}"

        actual_rsi = FeatureEngine._pipeline_compute_rsi(sample_close_prices_with_nan, period=period)

        pd.testing.assert_series_equal(actual_rsi, expected_rsi, check_dtype=True)
        assert actual_rsi.name == f"rsi_{period}"

    def test_rsi_input_not_series(self):
        result = FeatureEngine._pipeline_compute_rsi([10, 11, 12], period=5) # type: ignore
        assert isinstance(result, pd.Series)
        assert result.empty
        assert result.dtype == "float64"

# Tests for _pipeline_compute_macd (existing)
class TestPipelineComputeMACD:
    def test_macd_basic_calculation(self, sample_close_prices_long):
        fast = 12
        slow = 26
        signal = 9

        expected_macd_df = sample_close_prices_long.ta.macd(fast=fast, slow=slow, signal=signal)
        actual_macd_df = FeatureEngine._pipeline_compute_macd(sample_close_prices_long, fast=fast, slow=slow, signal=signal)
        pd.testing.assert_frame_equal(actual_macd_df, expected_macd_df, check_dtype=True)

    def test_macd_insufficient_data(self, sample_close_prices_short):
        fast = 12
        slow = 26
        signal = 9
        expected_macd_df = sample_close_prices_short.ta.macd(fast=fast, slow=slow, signal=signal)
        actual_macd_df = FeatureEngine._pipeline_compute_macd(sample_close_prices_short, fast=fast, slow=slow, signal=signal)
        pd.testing.assert_frame_equal(actual_macd_df, expected_macd_df, check_dtype=True)
        assert actual_macd_df.isna().all().all()

    def test_macd_with_nan_input(self, sample_close_prices_with_nan):
        fast = 12
        slow = 26
        signal = 9
        expected_macd_df = sample_close_prices_with_nan.ta.macd(fast=fast, slow=slow, signal=signal)
        actual_macd_df = FeatureEngine._pipeline_compute_macd(sample_close_prices_with_nan, fast=fast, slow=slow, signal=signal)
        pd.testing.assert_frame_equal(actual_macd_df, expected_macd_df, check_dtype=True)

    def test_macd_input_not_series(self):
        result = FeatureEngine._pipeline_compute_macd([10,11,12], fast=12, slow=26, signal=9) # type: ignore
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert result.dtypes.apply(lambda x: x == "float64").all()

# --- New Test Classes ---

class TestPipelineComputeBBands:
    def test_bbands_basic_calculation(self, sample_close_prices_long):
        length = 20
        std_dev = 2.0
        expected_df = sample_close_prices_long.ta.bbands(length=length, std=std_dev)
        actual_df = FeatureEngine._pipeline_compute_bbands(sample_close_prices_long, length=length, std_dev=std_dev)
        pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=True)

    def test_bbands_insufficient_data(self, sample_close_prices_short):
        length = 20
        std_dev = 2.0
        expected_df = sample_close_prices_short.ta.bbands(length=length, std=std_dev) # All NaNs
        actual_df = FeatureEngine._pipeline_compute_bbands(sample_close_prices_short, length=length, std_dev=std_dev)
        pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=True)
        assert actual_df.isna().all().all()

    def test_bbands_with_nan_input(self, sample_close_prices_with_nan):
        length = 10 # Shorter to ensure some calculation around NaNs
        std_dev = 2.0
        expected_df = sample_close_prices_with_nan.ta.bbands(length=length, std=std_dev)
        actual_df = FeatureEngine._pipeline_compute_bbands(sample_close_prices_with_nan, length=length, std_dev=std_dev)
        pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=True)

class TestPipelineComputeROC:
    def test_roc_basic_calculation(self, sample_close_prices_long):
        period = 10
        expected_series = sample_close_prices_long.ta.roc(length=period)
        expected_series.name = f"roc_{period}"
        actual_series = FeatureEngine._pipeline_compute_roc(sample_close_prices_long, period=period)
        pd.testing.assert_series_equal(actual_series, expected_series, check_dtype=True)
        assert actual_series.name == f"roc_{period}"

class TestPipelineComputeATR:
    def test_atr_basic_calculation(self, sample_ohlcv_data_long):
        length = 14
        expected_series = ta.atr(high=sample_ohlcv_data_long["high"], low=sample_ohlcv_data_long["low"], close=sample_ohlcv_data_long["close"], length=length)
        expected_series.name = f"atr_{length}"
        actual_series = FeatureEngine._pipeline_compute_atr(sample_ohlcv_data_long, length=length)
        pd.testing.assert_series_equal(actual_series, expected_series, check_dtype=True)
        assert actual_series.name == f"atr_{length}"

    def test_atr_missing_columns(self, sample_ohlcv_data_long):
        length = 14
        df_missing_high = sample_ohlcv_data_long.drop(columns=["high"])
        actual_series = FeatureEngine._pipeline_compute_atr(df_missing_high, length=length)
        assert actual_series.empty # Or specific error handling if implemented

class TestPipelineComputeStdev:
    def test_stdev_basic_calculation(self, sample_close_prices_long):
        length = 20
        expected_series = sample_close_prices_long.rolling(window=length).std()
        expected_series.name = f"stdev_{length}"
        actual_series = FeatureEngine._pipeline_compute_stdev(sample_close_prices_long, length=length)
        pd.testing.assert_series_equal(actual_series, expected_series, check_dtype=True)
        assert actual_series.name == f"stdev_{length}"

class TestPipelineComputeVWAP_OHLCV:
    def test_vwap_ohlcv_basic(self, sample_ohlcv_data_long):
        length = 14
        # Manual or trusted calculation for VWAP is complex for a rolling version.
        # Here, we mostly test structure and type.
        # The internal logic of _pipeline_compute_vwap_ohlcv was:
        # typical_price = (H+L+C)/3; (typical_price * V).rolling().sum() / V.rolling().sum()
        df = sample_ohlcv_data_long.copy()
        # Ensure inputs are Decimal for precision in intermediate calculations
        high_d = df["high"].apply(Decimal)
        low_d = df["low"].apply(Decimal)
        close_d = df["close"].apply(Decimal)
        volume_d = df["volume"].apply(Decimal)
        typical_price = (high_d + low_d + close_d) / Decimal("3.0")
        tp_vol = typical_price * volume_d
        sum_tp_vol = tp_vol.rolling(window=length, min_periods=length).sum()
        sum_vol = volume_d.rolling(window=length, min_periods=length).sum()
        expected_vwap_decimal = sum_tp_vol / sum_vol
        expected_vwap_float = expected_vwap_decimal.replace([Decimal("Infinity"), Decimal("-Infinity")], np.nan).astype("float64").fillna(np.nan)
        expected_vwap_float.name = f"vwap_ohlcv_{length}"

        actual_series = FeatureEngine._pipeline_compute_vwap_ohlcv(df, length=length) # df has float64
        pd.testing.assert_series_equal(actual_series, expected_vwap_float, check_dtype=True, atol=1e-9) # Added atol for float precision
        assert actual_series.name == f"vwap_ohlcv_{length}"

    def test_vwap_ohlcv_zero_volume(self, sample_ohlcv_data_long):
        length = 5
        df = sample_ohlcv_data_long.copy()
        df["volume"].iloc[0:length] = 0 # Zero volume for a window
        actual_series = FeatureEngine._pipeline_compute_vwap_ohlcv(df, length=length)
        # Expect NaN where sum_vol over window is 0
        assert actual_series.iloc[length-1:].isna().any() # Check that some NaNs are produced due to zero volume
        assert actual_series.name == f"vwap_ohlcv_{length}"


class TestPipelineL2Features:
    def test_l2_spread(self, sample_l2_books_series, sample_l2_book_valid):
        actual_df = FeatureEngine._pipeline_compute_l2_spread(sample_l2_books_series)
        assert isinstance(actual_df, pd.DataFrame)
        assert "abs_spread" in actual_df.columns
        assert "pct_spread" in actual_df.columns
        assert actual_df.index.equals(sample_l2_books_series.index)

        # Check valid book calculation
        valid_book = sample_l2_book_valid
        expected_abs = float(valid_book["asks"][0][0] - valid_book["bids"][0][0])
        expected_pct = float((expected_abs / float(valid_book["bids"][0][0] + valid_book["asks"][0][0])*Decimal("2")) * 100)
        assert np.isclose(actual_df["abs_spread"].iloc[0], expected_abs)
        assert np.isclose(actual_df["pct_spread"].iloc[0], expected_pct)
        assert actual_df["abs_spread"].iloc[1:].isna().all() # None and empty book

    def test_l2_imbalance(self, sample_l2_books_series, sample_l2_book_valid):
        levels = 2
        actual_series = FeatureEngine._pipeline_compute_l2_imbalance(sample_l2_books_series, levels=levels)
        assert isinstance(actual_series, pd.Series)
        assert actual_series.name == f"imbalance_{levels}"

        # Check valid book calculation
        valid_book = sample_l2_book_valid
        bid_vol = valid_book["bids"][0][1] + valid_book["bids"][1][1]
        ask_vol = valid_book["asks"][0][1] + valid_book["asks"][1][1]
        expected_imbalance = float((bid_vol - ask_vol) / (bid_vol + ask_vol))
        assert np.isclose(actual_series.iloc[0], expected_imbalance)
        assert actual_series.iloc[1] == np.nan # None book
        assert actual_series.iloc[2] == 0.0 # Empty book (or nan depending on impl choice)

    def test_l2_wap(self, sample_l2_books_series, sample_l2_book_valid):
        levels = 1
        actual_series = FeatureEngine._pipeline_compute_l2_wap(sample_l2_books_series, levels=levels)
        assert isinstance(actual_series, pd.Series)
        assert actual_series.name == f"wap_{levels}"

        valid_book = sample_l2_book_valid
        bbp = valid_book["bids"][0][0]; bbv = valid_book["bids"][0][1]
        bap = valid_book["asks"][0][0]; bav = valid_book["asks"][0][1]
        expected_wap = float((bbp * bav + bap * bbv) / (bbv + bav))
        assert np.isclose(actual_series.iloc[0], expected_wap)
        assert actual_series.iloc[1:].isna().all()


    def test_l2_depth(self, sample_l2_books_series, sample_l2_book_valid):
        levels = 2
        actual_df = FeatureEngine._pipeline_compute_l2_depth(sample_l2_books_series, levels=levels)
        assert isinstance(actual_df, pd.DataFrame)
        assert actual_df.columns.tolist() == [f"bid_depth_{levels}", f"ask_depth_{levels}"]

        valid_book = sample_l2_book_valid
        expected_bid_depth = float(valid_book["bids"][0][1] + valid_book["bids"][1][1])
        expected_ask_depth = float(valid_book["asks"][0][1] + valid_book["asks"][1][1])
        assert np.isclose(actual_df[f"bid_depth_{levels}"].iloc[0], expected_bid_depth)
        assert np.isclose(actual_df[f"ask_depth_{levels}"].iloc[0], expected_ask_depth)
        assert actual_df.iloc[1:].isna().all().all()


class TestPipelineTradeBasedFeatures:
    def test_vwap_trades(self, sample_trades_deque, sample_bar_start_times):
        interval = 60
        actual_series = FeatureEngine._pipeline_compute_vwap_trades(sample_trades_deque, sample_bar_start_times, interval)
        assert isinstance(actual_series, pd.Series)
        assert actual_series.name == f"vwap_trades_{interval}s"
        assert actual_series.index.equals(sample_bar_start_times.index)

        # Bar 1: (100.0*1.0 + 100.1*0.5 + 100.2*1.2) / (1.0+0.5+1.2) = 270.35 / 2.7 = 100.1296...
        # (Decimal('100.0') * Decimal('1.0') + Decimal('100.1') * Decimal('0.5') + Decimal('100.2') * Decimal('1.2')) / (Decimal('1.0') + Decimal('0.5') + Decimal('1.2'))
        expected_bar1_vwap = float(Decimal("270.35") / Decimal("2.7"))
        assert np.isclose(actual_series.iloc[0], expected_bar1_vwap)

        # Bar 2: (100.3*0.8 + 100.2*0.3) / (0.8+0.3) = (80.24 + 30.06) / 1.1 = 110.3 / 1.1 = 100.2727...
        expected_bar2_vwap = float(Decimal("110.3") / Decimal("1.1"))
        assert np.isclose(actual_series.iloc[1], expected_bar2_vwap)

        assert np.isnan(actual_series.iloc[2]) # No trades in bar 3

    def test_volume_delta(self, sample_trades_deque, sample_bar_start_times):
        interval = 60
        actual_series = FeatureEngine._pipeline_compute_volume_delta(sample_trades_deque, sample_bar_start_times, interval)
        assert isinstance(actual_series, pd.Series)
        assert actual_series.name == f"volume_delta_{interval}s"

        # Bar 1: buy(1.0) + sell(0.5) + buy(1.2) -> total_buy=2.2, total_sell=0.5. Delta = 2.2 - 0.5 = 1.7
        assert np.isclose(actual_series.iloc[0], 1.7)
        # Bar 2: buy(0.8) + sell(0.3) -> total_buy=0.8, total_sell=0.3. Delta = 0.8 - 0.3 = 0.5
        assert np.isclose(actual_series.iloc[1], 0.5)
        # Bar 3: No trades, should be 0.0 (as per current _pipeline_compute_volume_delta impl)
        assert np.isclose(actual_series.iloc[2], 0.0)

    def test_volume_delta_no_trades(self, sample_bar_start_times):
        interval = 60
        empty_deque = deque()
        actual_series = FeatureEngine._pipeline_compute_volume_delta(empty_deque, sample_bar_start_times, interval)
        assert actual_series.isna().all() # No trades in deque means NaN for all bars

    def test_vwap_trades_no_trades(self, sample_bar_start_times):
        interval = 60
        empty_deque = deque()
        actual_series = FeatureEngine._pipeline_compute_vwap_trades(empty_deque, sample_bar_start_times, interval)
        assert actual_series.isna().all()
