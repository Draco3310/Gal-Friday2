"""
Test configuration and shared fixtures for all tests.
"""
import os
import sys
import pytest

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture
def config_fixture():
    """
    Fixture that provides a test configuration.
    """
    return {
        "app_name": "Gal-Friday2",
        "environment": "test",
        "log_level": "INFO",
        "exchanges": {
            "kraken": {
                "api_key": "test_key",
                "api_secret": "test_secret"
            }
        },
        "database": {
            "connection_string": "sqlite:///:memory:"
        }
    }