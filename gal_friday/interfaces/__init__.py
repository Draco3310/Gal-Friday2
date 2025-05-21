"""
Core interfaces for the Gal-Friday trading system.

This module contains abstract base classes that define the essential interfaces
that various components of the system must implement.
"""

from .execution_handler_interface import ExecutionHandlerInterface
from .historical_data_service_interface import HistoricalDataService
from .market_price_service_interface import MarketPriceService
from .predictor_interface import PredictorInterface

__all__ = [
    "ExecutionHandlerInterface",
    "HistoricalDataService",
    "MarketPriceService",
    "PredictorInterface",
]
