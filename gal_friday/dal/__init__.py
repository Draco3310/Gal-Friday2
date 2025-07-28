"""Data Access Layer for Gal-Friday."""

from .base import BaseRepository  # Removed BaseEntity
from .connection_pool import DatabaseConnectionPool

# TODO: Resolve alembic autogen issue to allow direct import of TimeSeriesDB
# TimeSeriesDB import temporarily disabled for alembic autogen compatibility

__all__ = [
    "BaseRepository",  # Removed BaseEntity
    "DatabaseConnectionPool",
    # "TimeSeriesDB", # Commented out
]
