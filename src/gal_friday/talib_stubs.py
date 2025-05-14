"""Stub file for talib to satisfy imports in mypy checks.

This file provides minimal stub implementations for talib functions.
For actual functionality, the real talib package should be installed.
"""

import numpy as np


def RSI(close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index."""
    # This is just a stub
    return np.zeros(len(close))


def BBANDS(
    close: np.ndarray,
    timeperiod: int = 5,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
    matype: int = 0,
) -> tuple:
    """Calculate Bollinger Bands."""
    # This is just a stub
    return (np.zeros(len(close)), np.zeros(len(close)), np.zeros(len(close)))


def EMA(close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    # This is just a stub
    return np.zeros(len(close))


def SMA(close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """Calculate Simple Moving Average."""
    # This is just a stub
    return np.zeros(len(close))


def MACD(
    close: np.ndarray, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9
) -> tuple:
    """Calculate Moving Average Convergence/Divergence."""
    # This is just a stub
    return (np.zeros(len(close)), np.zeros(len(close)), np.zeros(len(close)))


def ATR(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """Calculate Average True Range."""
    # This is just a stub
    return np.zeros(len(close))
