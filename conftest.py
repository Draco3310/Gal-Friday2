"""
Root conftest.py for project-wide pytest configuration.

This file configures testing environment, registers custom markers, and sets up
fixtures that can be imported by all test modules.
"""
import os
from pathlib import Path
import sys

import pytest

# Add src to the Python path to ensure imports work correctly
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers.

    These markers can be used to categorize tests and selectively run them.
    """
    config.addinivalue_line("markers", "unit: mark a test as a unit test")
    config.addinivalue_line("markers", "integration: mark a test as an integration test")
    config.addinivalue_line("markers", "slow: mark a test as slow to run")
    config.addinivalue_line("markers", "network: mark a test that requires network access")
    config.addinivalue_line("markers", "exchange: mark a test that interacts with exchanges")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options to pytest."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests requiring network"
    )
    parser.addoption(
        "--run-exchange",
        action="store_true",
        default=False,
        help="run tests interacting with exchanges"
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item]
) -> None:
    """Skip tests based on markers and command line options."""
    # Skip slow tests unless --run-slow is provided
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip network tests unless --run-network is provided
    if not config.getoption("--run-network"):
        skip_network = pytest.mark.skip(reason="need --run-network option to run")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)

    # Skip exchange tests unless --run-exchange is provided
    if not config.getoption("--run-exchange"):
        skip_exchange = pytest.mark.skip(reason="need --run-exchange option to run")
        for item in items:
            if "exchange" in item.keywords:
                item.add_marker(skip_exchange)


@pytest.fixture(scope="session")
def test_env() -> None:
    """Set up the test environment variables."""
    # Save original environment variables
    original_env = dict(os.environ)

    # Set test environment variables
    os.environ["GAL_FRIDAY_ENV"] = "test"
    os.environ["GAL_FRIDAY_LOG_LEVEL"] = "ERROR"

    yield

    # Restore original environment variables
    os.environ.clear()
    os.environ.update(original_env)
