# Configuration Manager Module

import yaml
import os
import logging
from typing import Any, Optional, List, Dict
from functools import reduce
import operator
from decimal import Decimal

log = logging.getLogger(__name__)


class ConfigManager:
    """Manages loading and accessing application configuration from a YAML file."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        logger_service: Optional[logging.Logger] = None,
    ):
        """
        Initializes the ConfigManager, loads config, and validates it.

        Args:
            config_path: Path to the YAML configuration file relative to the workspace root.
            logger_service: Optional logger instance for dependency injection.
        """
        self._config_path = config_path
        self._config: Optional[dict] = None
        self.validation_errors: List[str] = []  # Initialize validation errors list

        # Use injected logger or default
        self._logger = logger_service or logging.getLogger(__name__)
        self._logger.info(f"Initializing ConfigManager with path: {self._config_path}")

        self.load_config()  # Load the configuration file into self._config

        # Validate the loaded configuration
        self.validation_errors = self.validate_configuration()

    def load_config(self) -> None:
        """Loads or reloads the configuration from the specified YAML file."""
        self._logger.info(f"Attempting to load configuration from: {self._config_path}")
        try:
            # Ensure the path is absolute or relative to the workspace root
            # Assuming the script runs from the workspace root for simplicity here.
            # A more robust solution might involve finding the project root
            # dynamically.
            if not os.path.exists(self._config_path):
                self._logger.error(f"Configuration file not found at: {self._config_path}")
                self._config = {}
                return

            with open(self._config_path, "r") as f:
                self._config = yaml.safe_load(f)
            self._logger.info(f"Successfully loaded configuration from {self._config_path}")
        except yaml.YAMLError as e:
            self._logger.exception(
                f"Error parsing YAML configuration file: {self._config_path}", exc_info=e
            )
            self._config = {}
        except IOError as e:
            self._logger.exception(
                f"Error reading configuration file: {self._config_path}", exc_info=e
            )
            self._config = {}
        except Exception as e:
            self._logger.exception(
                f"Error loading configuration from " f"{self._config_path}",
                exc_info=e,
            )
            self._config = {}

        if not isinstance(self._config, dict):
            self._logger.error(
                f"Configuration file {self._config_path} did not load as a dictionary. "
                f"Loaded type: {type(self._config)}. Setting config to empty dict."
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
            self._logger.warning(
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
            self._logger.debug(
                f"Key '{key}' not found in configuration. Returning default: {default}"
            )
            return default
        except Exception as e:
            self._logger.exception(
                f"Unexpected error retrieving key '{key}' from configuration.", exc_info=e
            )
            return default

    def get_int(self, key: str, default: int = 0) -> int:
        """Retrieves a config value and attempts to cast it to an integer."""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError) as e:
            self._logger.warning(
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
            self._logger.warning(
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
                self._logger.warning(
                    f"Invalid default value '{default}' for get_decimal, using 0.0"
                )
                default = Decimal("0.0")

        value = self.get(key, default)
        try:
            # Convert to string first to handle floats/ints correctly
            return Decimal(str(value))
        except Exception as e:
            self._logger.warning(
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
            if val_lower in ["true", "yes", "1"]:
                return True
            if val_lower in ["false", "no", "0"]:
                return False
        if isinstance(value, (int, float)):
            return value != 0
        if value is None:
            return False  # Treat None as False

        # If not interpretable, return default
        self._logger.warning(
            f"Could not interpret value for key '{key}' ('{value}') as bool. "
            f"Returning default {default}."
        )
        return default

    def get_list(self, key: str, default: Optional[List[Any]] = None) -> List[Any]:
        """Retrieves a config value expected to be a list."""
        if default is None:
            default = []  # Default to empty list if None specified

        value = self.get(key, default)
        if isinstance(value, list):
            return value
        else:
            self._logger.warning(
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
            self._logger.warning(
                f"Value for key '{key}' is not a dict (type: {type(value)}). "
                f"Returning default {default}."
            )
            # Ensure the default is returned if the fetched value wasn't a dict
            return default if isinstance(default, dict) else {}

    def validate_configuration(self) -> List[str]:
        """
        Validates the loaded configuration against predefined rules.
        Returns a list of validation error messages. An empty list indicates success.
        """
        errors: List[str] = []

        if self._config is None:
            errors.append("Internal error: Configuration object is None.")
            return errors

        if not self._config:
            errors.append("Configuration is empty. Check config file path and format.")
            return errors

        # --- Validation Checks ---

        # 1. Check required top-level sections
        required_top_level = ["trading", "risk", "api"]
        for section in required_top_level:
            if self.get(section) is None:
                errors.append(f"Missing required configuration section: '{section}'")

        # 2. Validate 'trading' section (only if it exists)
        self._validate_trading_section(errors)

        # 3. Validate 'risk' section (only if it exists)
        self._validate_risk_section(errors)

        # 4. Validate 'api' section (only if it exists)
        self._validate_api_section(errors)

        # --- Logging ---
        if errors:
            self._logger.error("Configuration validation failed with %d error(s):", len(errors))
            for error in errors:
                self._logger.error("- %s", error)
        else:
            self._logger.info("Configuration validation successful.")

        return errors

    def _is_valid_trading_pair(self, pair: str) -> bool:
        """Validates if a string is a properly formatted trading pair."""
        if not isinstance(pair, str):
            return False
        if "/" not in pair:
            return False
        parts = pair.split("/")
        if len(parts) != 2:
            return False
        if not all(p.strip() for p in parts):
            return False
        return True

    def _validate_trading_section(self, errors: List[str]) -> None:
        """Validates the 'trading' section of the configuration."""
        if self.get("trading") is None:
            return
        if not isinstance(self.get("trading"), dict):
            errors.append("'trading' section must be a dictionary.")
            return
        # Validate trading pairs
        pairs = self.get_list("trading.pairs", None)
        if pairs is None:
            errors.append("Missing required key: 'trading.pairs'")
        elif not pairs:
            errors.append("Configuration key 'trading.pairs' cannot be empty.")
        else:
            # Validate format of each pair
            for i, pair in enumerate(pairs):
                if not self._is_valid_trading_pair(pair):
                    errors.append(
                        f"Invalid trading pair format at index {i}: '{pair}'. "
                        f"Expected 'BASE/QUOTE' (e.g., 'BTC/USD')."
                    )
        # Validate exchange
        exchange = self.get("trading.exchange")
        if exchange is None:
            errors.append("Missing required key: 'trading.exchange'")
        elif not isinstance(exchange, str) or not exchange.strip():
            errors.append("'trading.exchange' must be a non-empty string.")

    def _validate_risk_section(self, errors: List[str]) -> None:
        """Validates the 'risk' section of the configuration."""
        if self.get("risk") is None:
            return
        if not isinstance(self.get("risk"), dict):
            errors.append("'risk' section must be a dictionary.")
            return
        # Validate max_drawdown_pct
        max_drawdown = self.get_float("risk.max_drawdown_pct", 0.0)
        if max_drawdown == 0.0 and self.get("risk.max_drawdown_pct") is None:
            errors.append("Missing required key: 'risk.max_drawdown_pct'")
        elif max_drawdown <= 0:
            errors.append("'risk.max_drawdown_pct' must be a positive value.")
        # Example cross-validation:
        stop_loss = self.get_float("risk.stop_loss_pct", 0.0)
        take_profit = self.get_float("risk.take_profit_pct", 0.0)
        # Only validate if both values are explicitly set (not using defaults)
        if (
            self.get("risk.stop_loss_pct") is not None
            and self.get("risk.take_profit_pct") is not None
            and stop_loss >= take_profit
        ):
            errors.append("'risk.stop_loss_pct' must be less than 'risk.take_profit_pct'.")

    def _validate_api_section(self, errors: List[str]) -> None:
        """Validates the 'api' section of the configuration."""
        if self.get("api") is None:
            return
        if not isinstance(self.get("api"), dict):
            errors.append("'api' section must be a dictionary.")
            return
        # Check if at least one service (e.g., kraken, binance) is configured
        if not self.get_dict("api"):  # Check if the api dict itself is empty
            errors.append(
                "'api' section cannot be empty. Configure at least one service (e.g., 'kraken')."
            )
            return
        # Validate specific services if needed (e.g., ensure kraken has key/secret)
        for service_name in self.get_dict("api").keys():
            # Use the secure getters which check env vars first
            api_key = self.get_secure_api_key(service_name)
            api_secret = self.get_secure_api_secret(service_name)
            if api_key is None:
                errors.append(
                    f"Missing API key for service '{service_name}'. Check "
                    f"'api.{service_name}.key' in config or the "
                    f"'{service_name.upper()}_KEY' env var."
                )
            if api_secret is None:
                errors.append(
                    f"Missing API secret for service '{service_name}'. Check "
                    f"'api.{service_name}.secret' in config or the "
                    f"'{service_name.upper()}_SECRET' env var."
                )

    # --- Optional specific getters (can be added as needed) ---

    def get_trading_pairs(self) -> List[str]:
        """Retrieves the list of trading pairs."""
        # Validation happens in validate_configuration
        pairs = self.get_list("trading.pairs", [])
        return pairs

    def get_risk_parameters(self) -> Dict[str, Any]:
        """Retrieves the risk configuration section."""
        # Validation happens in validate_configuration
        return self.get_dict("risk", {})

    def get_strategy_parameters(self, strategy_id: str) -> Dict[str, Any]:
        """Retrieves parameters for a specific strategy."""
        # Validation happens in validate_configuration
        return self.get_dict(f"strategies.{strategy_id}", {})

    def get_api_keys(self, service_name: str) -> Dict[str, Optional[str]]:
        """
        Retrieves API key/secret pair securely for a given service.
        Assumes standard key names 'key' and 'secret'.
        Returns a dict with 'key' and 'secret' containing Optional[str].
        """
        self._logger.info(f"Retrieving secure API credentials for service: {service_name}")
        return {
            "key": self.get_secure_api_key(service_name, "key"),
            "secret": self.get_secure_api_secret(service_name, "secret"),
            # Add other potential credential fields if needed, e.g., 'password'
            # 'password': self.get_secure_value(f'api.{service_name}.password')
        }

    def reload_config(self) -> List[str]:
        """
        Reloads the configuration from the file and re-validates it.

        Updates `self.validation_errors` with the results of the new validation.

        Returns:
            List of validation errors encountered during the reload and validation process.
            An empty list indicates the reload and validation were successful.
        """
        self._logger.info(f"Attempting to reload configuration from: {self._config_path}")

        self.load_config()  # Reloads self._config. Handles file read errors internally.

        # Re-validate the newly loaded configuration
        self.validation_errors = self.validate_configuration()

        if not self.validation_errors:
            self._logger.info("Configuration reloaded and validated successfully.")
        else:
            # Errors already logged by validate_configuration
            self._logger.warning("Configuration reload completed, but validation failed.")

        return self.validation_errors

    def is_valid(self) -> bool:
        """Returns True if configuration was loaded successfully AND passed validation."""
        # Check if config was loaded (is not None) and if there are no validation errors.
        return self._config is not None and not self.validation_errors

    def get_secure_value(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieves sensitive configuration values, prioritizing environment variables.
        Converts dot notation key to uppercase underscore notation for env var lookup.
        Example: 'api.kraken.key' becomes 'API_KRAKEN_KEY'.
        Logs source (env or config) but not the value itself unless default is returned.
        """
        env_var_name = key.replace(".", "_").upper()
        env_value = os.environ.get(env_var_name)

        if env_value is not None:
            # Log that it was found in env, but not the value
            self._logger.info(
                f"Retrieved secure value for '{key}' from environment variable '{env_var_name}'."
            )
            return env_value
        else:
            # Fall back to config file using the regular 'get' method
            config_value = self.get(key, default)  # 'get' handles logging for missing keys
            if config_value is not None and config_value != default:
                # Log that it was found in config, but not the value
                self._logger.debug(
                    f"Retrieved secure value for '{key}' from config file "
                    f"(env var '{env_var_name}' not set)."
                )
            elif config_value is None and default is None:
                # Log warning only if it's truly missing (not just using default=None)
                self._logger.warning(
                    f"Secure value for '{key}' not found in environment or config file. "
                    f"Returning None."
                )

            # Ensure we return an Optional[str] to satisfy mypy
            if config_value is None:
                return None
            elif isinstance(config_value, str):
                return config_value
            else:
                # Convert non-string values to strings
                self._logger.debug(f"Converting non-string config value for '{key}' to string.")
                return str(config_value)

    def get_secure_api_key(self, service_name: str, key_name: str = "key") -> Optional[str]:
        """Retrieves a specific API key securely for a given service."""
        full_key = f"api.{service_name}.{key_name}"
        return self.get_secure_value(full_key)

    def get_secure_api_secret(
        self, service_name: str, secret_name: str = "secret"
    ) -> Optional[str]:
        """Retrieves a specific API secret securely for a given service."""
        full_key = f"api.{service_name}.{secret_name}"
        return self.get_secure_value(full_key)


# Example Usage (for testing purposes, remove in production)
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a dummy config file for testing
    dummy_config_dir = "config"
    dummy_config_path = os.path.join(dummy_config_dir, "test_config.yaml")
    if not os.path.exists(dummy_config_dir):
        os.makedirs(dummy_config_dir)

    # Example configuration with all required sections
    dummy_data = {
        "trading": {"pairs": ["XRP/USD", "DOGE/USD"], "exchange": "kraken"},
        "database": {
            "postgres": {"host": "localhost", "port": 5432, "user": "galfriday_user"},
            "influxdb": {"url": "http://localhost:8086", "token": "YOUR_INFLUX_TOKEN"},
        },
        "risk": {
            "max_drawdown_pct": 5.0,  # Added required field
            "stop_loss_pct": 2.0,  # Added for cross-validation
            "take_profit_pct": 5.0,  # Added for cross-validation
            "limits": {"max_total_drawdown_pct": 10.0},
        },
        "monitoring": {"check_interval_seconds": 30},
        "api": {"kraken": {"key": "YOUR_API_KEY_HERE", "secret": "YOUR_API_SECRET_HERE"}},
        "strategies": {"momentum": {"lookback_periods": 14, "threshold": 0.02}},
    }

    with open(dummy_config_path, "w") as f:
        yaml.dump(dummy_data, f, default_flow_style=False)

    print(f"Created dummy config at: {dummy_config_path}")

    # Initialize ConfigManager with the dummy file
    print("\n=== BASIC CONFIGURATION LOADING ===")
    config_manager = ConfigManager(config_path=dummy_config_path)

    # Check if configuration is valid
    print(f"Configuration valid: {config_manager.is_valid()}")

    if config_manager.validation_errors:
        print("Validation errors:")
        for error in config_manager.validation_errors:
            print(f"  - {error}")

    # Test retrieving values
    print("\n=== TESTING GETTERS ===")
    print(f"Exchange: {config_manager.get('trading.exchange')}")
    print(f"Postgres Host: {config_manager.get('database.postgres.host')}")
    print(f"Max Drawdown %: {config_manager.get_float('risk.max_drawdown_pct')}")
    print(f"Monitoring Interval: {config_manager.get_int('monitoring.check_interval_seconds')}")

    # Test interface methods
    print("\n=== TESTING INTERFACE METHODS ===")
    print(f"Trading Pairs: {config_manager.get_trading_pairs()}")

    risk_params = config_manager.get_risk_parameters()
    print(f"Risk Parameters: {risk_params}")

    strategy_params = config_manager.get_strategy_parameters("momentum")
    print(f"Momentum Strategy Parameters: {strategy_params}")

    # Test secure API handling - This will show masked values in the output
    # In a real app, this would be retrieved from environment variables
    print("\n=== TESTING SECURE API HANDLING ===")
    api_keys = config_manager.get_api_keys("kraken")
    # Safely display secrets by masking them
    key = api_keys.get("key")
    secret = api_keys.get("secret")
    masked_api_keys = {
        "key": "xxxx" + key[-4:] if key else None,
        "secret": "xxxx" + secret[-4:] if secret else None,
    }
    print(f"Kraken API Keys (masked): {masked_api_keys}")

    print("\n=== TESTING CONFIG RELOAD ===")
    # Modify the config file to introduce a validation error
    print("Modifying config to introduce an error...")
    # Use a properly typed dictionary for mypy
    trading_section = dummy_data.get("trading", {})
    if isinstance(trading_section, dict):
        trading_section["pairs"] = ["Invalid_Pair_Format"]  # Will fail validation
    with open(dummy_config_path, "w") as f:
        yaml.dump(dummy_data, f, default_flow_style=False)

    # Reload configuration
    reload_errors = config_manager.reload_config()
    print(f"Configuration valid after reload: {config_manager.is_valid()}")

    if reload_errors:
        print("Reload validation errors:")
        for error in reload_errors:
            print(f"  - {error}")

    # Fix the error and reload again
    print("\nFixing the error and reloading...")
    # Use a properly typed dictionary for mypy
    trading_section = dummy_data.get("trading", {})
    if isinstance(trading_section, dict):
        trading_section["pairs"] = ["XRP/USD", "DOGE/USD"]  # Fixed format
    with open(dummy_config_path, "w") as f:
        yaml.dump(dummy_data, f, default_flow_style=False)

    reload_errors = config_manager.reload_config()
    print(f"Configuration valid after fixing: {config_manager.is_valid()}")

    # Clean up dummy file (commented out for inspection)
    # os.remove(dummy_config_path)
    # os.rmdir(dummy_config_dir) # Only if empty
    print(f"\n(Remember to manually delete {dummy_config_path} and {dummy_config_dir} if needed)")
