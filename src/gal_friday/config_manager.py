# Configuration Manager Module

import yaml
import os
import logging
from typing import Any, Optional, List
from functools import reduce
import operator
from decimal import Decimal

log = logging.getLogger(__name__)


class ConfigManager:
    """Manages loading and accessing application configuration from a YAML file."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initializes the ConfigManager.

        Args:
            config_path: Path to the YAML configuration file relative to the workspace root.
        """
        self._config_path = config_path
        self._config: Optional[dict] = None
        self.load_config()

    def load_config(self) -> None:
        """Loads or reloads the configuration from the specified YAML file."""
        log.info(f"Attempting to load configuration from: {self._config_path}")
        try:
            # Ensure the path is absolute or relative to the workspace root
            # Assuming the script runs from the workspace root for simplicity here.
            # A more robust solution might involve finding the project root dynamically.
            if not os.path.exists(self._config_path):
                log.error(
                    f"Configuration file not found at: {self._config_path}"
                )
                self._config = {}
                return

            with open(self._config_path, "r") as f:
                self._config = yaml.safe_load(f)
            log.info(
                f"Successfully loaded configuration from {self._config_path}"
            )
        except yaml.YAMLError as e:
            log.exception(
                "Error parsing YAML configuration file: "
                f"{self._config_path}",
                exc_info=e
            )
            self._config = {}
        except IOError as e:
            log.exception(
                "Error reading configuration file: "
                f"{self._config_path}",
                exc_info=e
            )
            self._config = {}
        except Exception as e:
            log.exception(
                (
                    "An unexpected error occurred while loading configuration "
                    f"from {self._config_path}"
                ),
                exc_info=e,
            )
            self._config = {}

        if not isinstance(self._config, dict):
            log.error(
                (
                    f"Configuration file {self._config_path} did not load as a dictionary. "
                    f"Loaded type: {type(self._config)}. Setting config to empty dict."
                )
            )
            self._config = {}

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieves a configuration value using a dot-separated key.

        Example:
            config.get('database.postgres.host', 'localhost')

        Args:
            key: The dot-separated key string.
            default: The value to return if the key is not found.

        Returns:
            The configuration value or the default.
        """
        if self._config is None:
            log.warning(
                "Configuration accessed before it was loaded or after a loading error."
            )
            return default

        try:
            # Use reduce to navigate the nested dictionary structure
            value = reduce(operator.getitem, key.split("."), self._config)
            return value
        except (KeyError, TypeError):
            # KeyError if a key in the path doesn't exist
            # TypeError if trying to index into a non-dictionary
            log.debug(
                f"Key '{key}' not found in configuration. "
                f"Returning default: {default}"
            )
            return default
        except Exception as e:
            log.exception(
                f"Unexpected error retrieving key '{key}' from configuration.",
                exc_info=e
            )
            return default

    def get_int(self, key: str, default: int = 0) -> int:
        """Retrieves a config value and attempts to cast it to an integer."""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError) as e:
            log.warning(
                f"Could not convert value for key '{key}' ('{value}') to int. "
                f"Returning default {default}. Error: {e}"
            )
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Retrieves a config value and attempts to cast it to a float."""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            log.warning(
                f"Could not convert value for key '{key}' ('{value}') to float. "
                f"Returning default {default}. Error: {e}"
            )
            return default

    def get_decimal(self, key: str, default: Decimal = Decimal("0.0")) -> Decimal:
        """Retrieves a config value and attempts to cast it to a Decimal."""
        # Ensure default is Decimal if provided otherwise
        if not isinstance(default, Decimal):
            try:
                default = Decimal(str(default))
            except Exception:
                log.warning(f"Invalid default value '{default}' for get_decimal, using 0.0")
                default = Decimal("0.0")
                
        value = self.get(key, default)
        try:
            # Convert to string first to handle floats/ints correctly
            return Decimal(str(value))
        except Exception as e:
            log.warning(
                f"Could not convert value for key '{key}' ('{value}') to Decimal. "
                f"Returning default {default}. Error: {e}"
            )
            return default
            
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Retrieves a config value and attempts to interpret it as a boolean.
           Considers true: 'true', 'yes', '1', True (case-insensitive).
           Considers false: 'false', 'no', '0', False, None (case-insensitive).
        """
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            val_lower = value.lower()
            if val_lower in ['true', 'yes', '1']:
                return True
            if val_lower in ['false', 'no', '0']:
                return False
        if isinstance(value, (int, float)):
            return value != 0
        if value is None:
             return False # Treat None as False
             
        # If not interpretable, return default
        log.warning(
            f"Could not interpret value for key '{key}' ('{value}') as bool. "
            f"Returning default {default}."
        )
        return default

    def get_list(self, key: str, default: Optional[List[Any]] = None) -> List[Any]:
        """Retrieves a config value expected to be a list."""
        if default is None:
            default = [] # Default to empty list if None specified
            
        value = self.get(key, default)
        if isinstance(value, list):
            return value
        else:
            log.warning(
                f"Value for key '{key}' is not a list (type: {type(value)}). "
                f"Returning default {default}."
            )
            # Ensure the default is returned if the fetched value wasn't a list
            return default if isinstance(default, list) else [] 

    def get_dict(self, key: str, default: Optional[dict] = None) -> dict:
        """Retrieves a config value expected to be a dictionary."""
        if default is None:
            default = {}  # Default to empty dict if None specified
            
        value = self.get(key, default)
        if isinstance(value, dict):
            return value
        else:
            log.warning(
                f"Value for key '{key}' is not a dict (type: {type(value)}). "
                f"Returning default {default}."
            )
            # Ensure the default is returned if the fetched value wasn't a dict
            return default if isinstance(default, dict) else {}

    # --- Optional specific getters (can be added as needed) ---

    # def get_trading_pairs(self) -> list[str]:
    #     return self.get('trading.pairs', [])

    # def get_risk_parameters(self) -> dict:
    #     return self.get('risk', {})

    # def get_db_config(self, db_type: str = 'postgres') -> dict:
    #     return self.get(f'database.{db_type}', {})


# Example Usage (for testing purposes, remove in production)
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a dummy config file for testing
    dummy_config_dir = "config"
    dummy_config_path = os.path.join(dummy_config_dir, "test_config.yaml")
    if not os.path.exists(dummy_config_dir):
        os.makedirs(dummy_config_dir)

    dummy_data = {
        "trading": {"pairs": ["XRP/USD", "DOGE/USD"], "exchange": "kraken"},
        "database": {
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "user": "galfriday_user"
            },
            "influxdb": {
                "url": "http://localhost:8086",
                "token": "YOUR_INFLUX_TOKEN"
            },
        },
        "risk": {"limits": {"max_total_drawdown_pct": 10.0}},
        "monitoring": {"check_interval_seconds": 30},
    }
    with open(dummy_config_path, "w") as f:
        yaml.dump(dummy_data, f, default_flow_style=False)

    print(f"Created dummy config at: {dummy_config_path}")

    # Initialize ConfigManager with the dummy file
    config_manager = ConfigManager(config_path=dummy_config_path)

    # Test retrieving values
    print(f"Exchange: {config_manager.get('trading.exchange')}")
    print(f"Postgres Host: {config_manager.get('database.postgres.host')}")
    print(
        f"Max Drawdown %: "
        f"{config_manager.get('risk.limits.max_total_drawdown_pct')}"
    )
    print(
        f"Non-existent key (with default): "
        f"{config_manager.get('api.kraken.secret', 'DEFAULT_SECRET')}"
    )
    print(f"Non-existent key (no default): {config_manager.get('some.other.key')}")
    print(
        f"Monitoring Interval: "
        f"{config_manager.get('monitoring.check_interval_seconds')}"
    )

    # Clean up dummy file
    # os.remove(dummy_config_path)
    # os.rmdir(dummy_config_dir) # Only if empty
    print(
        f"(Remember to manually delete {dummy_config_path} and "
        f"{dummy_config_dir} if needed)"
    )
