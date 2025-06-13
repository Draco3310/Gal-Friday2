"""Monitoring module for Gal-Friday trading system.

This module provides dashboard, alerting, and live data collection functionality.
"""

from .dashboard_backend import initialize_dashboard
from .live_data_collector import LiveDataCollector

__all__ = [
    "initialize_dashboard",
    "LiveDataCollector",
]