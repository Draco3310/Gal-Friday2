"""Tests for the config_manager module."""

from gal_friday.config_manager import ConfigManager


def test_config_manager_initialization():
    """Test that the ConfigManager initializes correctly."""
    config = ConfigManager("config/config.yaml")
    assert config is not None

    # Test with config fixture
    config = ConfigManager(config_dict={"app_name": "test", "environment": "test"})
    assert config.get("app_name") == "test"
    assert config.get("environment") == "test"


def test_config_manager_get_method():
    """Test that the get method returns the correct values."""
    config = ConfigManager(config_dict={"app_name": "Gal-Friday2", "nested": {"key": "value"}})

    assert config.get("app_name") == "Gal-Friday2"
    assert config.get("nested.key") == "value"
    assert config.get("non_existent", default="default") == "default"


def test_config_manager_reload():
    """Test config reload functionality."""
    config = ConfigManager(config_dict={"version": "1.0"})
    assert config.get("version") == "1.0"

    # Test reload with new values
    config.reload({"version": "2.0"})
    assert config.get("version") == "2.0"
