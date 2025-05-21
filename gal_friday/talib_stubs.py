"""Stub file for talib to satisfy imports in mypy checks.

This file provides minimal stub implementations for talib functions.
For actual functionality, the real talib package should be installed.
"""

import numpy as np


def rsi(close: np.ndarray, _timeperiod: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index."""
    # This is just a stub
    return np.zeros(len(close))


def bbands(
    close: np.ndarray,
    _timeperiod: int = 5,
    _nbdevup: float = 2.0,
    _nbdevdn: float = 2.0,
    _matype: int = 0,
) -> tuple:
    """Calculate Bollinger Bands."""
    # This is just a stub
    return (np.zeros(len(close)), np.zeros(len(close)), np.zeros(len(close)))


def ema(close: np.ndarray, _timeperiod: int = 30) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    # This is just a stub
    return np.zeros(len(close))


def sma(close: np.ndarray, _timeperiod: int = 30) -> np.ndarray:
    """Calculate Simple Moving Average."""
    # This is just a stub
    return np.zeros(len(close))


def macd(
    close: np.ndarray, _fastperiod: int = 12, _slowperiod: int = 26, _signalperiod: int = 9
) -> tuple:
    """Calculate Moving Average Convergence/Divergence."""
    # This is just a stub
    return (np.zeros(len(close)), np.zeros(len(close)), np.zeros(len(close)))


def atr(
    _high: np.ndarray,
    _low: np.ndarray,
    close: np.ndarray,
    _timeperiod: int = 14,
) -> np.ndarray:
    """Calculate Average True Range."""
    # This is just a stub
    return np.zeros(len(close))
