"""Base model module for SQLAlchemy declarative base.

This module provides the base class for all database models
in the Gal-Friday trading system.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
