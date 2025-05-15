"""Provide configuration management for the Gal-Friday trading system.

This module handles loading, validating, and accessing application configuration from
YAML files. It provides secure access to sensitive information like API keys and supports
both file-based configuration and environment variable overrides.
"""

# Configuration Manager Module

from decimal import Decimal
from functools import reduce
import logging
import operator
import os
from pathlib import Path  # Added for PTH110 and PTH123
from typing import Any, Optional

import yaml

log = logging.getLogger(__name__)


class ConfigManager:
    """Manage loading and accessing application configuration from a YAML file."""

    _EXPECTED_TRADING_PAIR_COMPONENTS = 2

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        logger_service: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the ConfigManager, load config, and validate it.

        Args
        ----
            config_path: Path to the YAML configuration file relative to the workspace root.
            logger_service: Optional logger instance for dependency injection.
        """
        self._config_path = config_path
        self._config: Optional[dict] = None
        self.validation_errors: list[str] = []  # Initialize validation errors list

        # Use injected logger or default
        self._logger = logger_service or logging.getLogger(__name__)
        self._logger.info("Initializing ConfigManager with path: %s", self._config_path)

        self.load_config()  # Load the configuration file into self._config

        # Validate the loaded configuration
        self.validation_errors = self.validate_configuration()

    def load_config(self) -> None:
        """Load or reload the configuration from the specified YAML file."""
        self._logger.info("Attempting to load configuration from: %s", self._config_path)
        try:
            # Ensure the path is absolute or relative to the workspace root
            # Assuming the script runs from the workspace root for simplicity here.
            # A more robust solution might involve finding the project root
            # dynamically.
            config_file = Path(self._config_path)  # Use Path object
            if not config_file.exists():  # PTH110 fix
                self._logger.error("Configuration file not found at: %s", self._config_path)
                self._config = {}
                return

            with config_file.open() as f:  # PTH123 fix
                self._config = yaml.safe_load(f)
            self._logger.info("Successfully loaded configuration from %s", self._config_path)
        except yaml.YAMLError as e:
            self._logger.exception(
                "Error parsing YAML configuration file: %s", self._config_path, exc_info=e
            )
            self._config = {}
        except OSError as e:
            self._logger.exception(
                "Error reading configuration file: %s", self._config_path, exc_info=e
            )
            self._config = {}
        except Exception as e:
            self._logger.exception(
                "Error loading configuration from %s", self._config_path,
                exc_info=e,
            )
            self._config = {}

        if not isinstance(self._config, dict):
            self._logger.error(
                "Configuration file %s did not load as a dictionary. "
                "Loaded type: %s. Setting config to empty dict.",
                self._config_path,
                type(self._config),
            )
            self._config = {}

    def get(self, key: str, default: Optional[Any] = None) -> Any:  # noqa: ANN401
        """
        Retrieve a configuration value using a dot-separated key.

        Example:
            config.get('database.postgres.host', 'localhost')

        Args
        ----
            key: The dot-separated key string.
            default: The value to return if the key is not found.

        Returns
        -------
            The configuration value or the default.
        """
        if self._config is None:
            self._logger.warning(
                "Configuration accessed before it was loaded or after a loading error."
            )
            return default

        try:
            pass  # Logic is in the else block
        except (KeyError, TypeError):
            # KeyError if a key in the path doesn't exist
            # TypeError if trying to index into a non-dictionary
            self._logger.debug(
                "Key '%s' not found in configuration. Returning default: %s", key, default
            )
            return default
        except Exception as e:
            self._logger.exception(
                "Unexpected error retrieving key '%s' from configuration.", key, exc_info=e
            )
            return default
        else:
            # This else block executes if the try block completes without an exception.
            return reduce(operator.getitem, key.split("."), self._config)

    def get_int(self, key: str, default: int = 0) -> int:
        """Retrieve a config value and attempt to cast it to an integer."""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError) as e:
            self._logger.warning(
                "Could not convert value for key '%s' ('%s') to int. "
                "Returning default %s. Error: %s",
                key,
                value,
                default,
                e,
            )
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Retrieve a config value and attempt to cast it to a float."""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            self._logger.warning(
                "Could not convert value for key '%s' ('%s') to float. "
                "Returning default %s. Error: %s",
                key,
                value,
                default,
                e,
            )
            return default

    def get_decimal(self, key: str, default: Decimal = Decimal("0.0")) -> Decimal:
        """Retrieve a config value and attempt to cast it to a Decimal."""
        # Ensure default is Decimal if provided otherwise
        if not isinstance(default, Decimal):
            try:
                default = Decimal(str(default))
            except Exception:
                self._logger.warning(
                    "Invalid default value '%s' for get_decimal, using 0.0", default
                )
                default = Decimal("0.0")

        value = self.get(key, default)
        try:
            # Convert to string first to handle floats/ints correctly
            return Decimal(str(value))
        except Exception as e:
            self._logger.warning(
                "Could not convert value for key '%s' ('%s') to Decimal. "
                "Returning default %s. Error: %s",
                key,
                value,
                default,
                e,
            )
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Retrieve a config value and attempt to interpret it as a boolean.

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
            "Could not interpret value for key '%s' ('%s') as bool. "
            "Returning default %s.",
            key,
            value,
            default,
        )
        return default

    def get_list(self, key: str, default: Optional[list[Any]] = None) -> list[Any]:
        """Retrieve a config value expected to be a list."""
        if default is None:
            default = []  # Default to empty list if None specified

        value = self.get(key, default)
        if isinstance(value, list):
            return value
        self._logger.warning(
            "Value for key '%s' is not a list (type: %s). "
            "Returning default %s.",
            key,
            type(value),
            default,
        )
        # Ensure the default is returned if the fetched value wasn't a list
        return default if isinstance(default, list) else []

    def get_dict(self, key: str, default: Optional[dict] = None) -> dict:
        """Retrieve a config value expected to be a dictionary."""
        if default is None:
            default = {}  # Default to empty dict if None specified

        value = self.get(key, default)
        if isinstance(value, dict):
            return value
        self._logger.warning(
            "Value for key '%s' is not a dict (type: %s). "
            "Returning default %s.",
            key,
            type(value),
            default,
        )
        # Ensure the default is returned if the fetched value wasn't a dict
        return default if isinstance(default, dict) else {}

    def validate_configuration(self) -> list[str]:
        """
        Validate the loaded configuration against predefined rules.

        Returns a list of validation error messages. An empty list indicates success.
        """
        errors: list[str] = []

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
        """Validate if a string is a properly formatted trading pair."""
        if not isinstance(pair, str):
            return False
        if "/" not in pair:
            return False
        parts = pair.split("/")
        if len(parts) != self._EXPECTED_TRADING_PAIR_COMPONENTS:
            return False
        return all(p.strip() for p in parts)

    def _validate_trading_section(self, errors: list[str]) -> None:
        """Validate the 'trading' section of the configuration."""
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

    def _validate_risk_section(self, errors: list[str]) -> None:
        """Validate the 'risk' section of the configuration."""
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

    def _validate_api_section(self, errors: list[str]) -> None:
        """Validate the 'api' section of the configuration."""
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
        for service_name in self.get_dict("api"):
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

    def get_trading_pairs(self) -> list[str]:
        """Retrieve the list of trading pairs."""
        # Validation happens in validate_configuration
        return self.get_list("trading.pairs", [])

    def get_risk_parameters(self) -> dict[str, Any]:
        """Retrieve the risk configuration section."""
        # Validation happens in validate_configuration
        return self.get_dict("risk", {})

    def get_strategy_parameters(self, strategy_id: str) -> dict[str, Any]:
        """Retrieve parameters for a specific strategy."""
        # Validation happens in validate_configuration
        return self.get_dict(f"strategies.{strategy_id}", {})

    def get_api_keys(self, service_name: str) -> dict[str, Optional[str]]:
        """
        Retrieve API key/secret pair securely for a given service.

        Assumes standard key names 'key' and 'secret'.
        Returns a dict with 'key' and 'secret' containing Optional[str].
        """
        self._logger.info("Retrieving secure API credentials for service: %s", service_name)
        return {
            "key": self.get_secure_api_key(service_name, "key"),
            "secret": self.get_secure_api_secret(service_name, "secret"),
            # Add other potential credential fields if needed, e.g., 'password'
            # 'password': self.get_secure_value(f'api.{service_name}.password')
        }

    def reload_config(self) -> list[str]:
        """
        Reload the configuration from the file and re-validate it.

        Updates `self.validation_errors` with the results of the new validation.

        Returns
        -------
            List of validation errors encountered during the reload and validation process.
            An empty list indicates the reload and validation were successful.
        """
        self._logger.info("Attempting to reload configuration from: %s", self._config_path)

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
        """Return True if configuration was loaded successfully AND passed validation."""
        # Check if config was loaded (is not None) and if there are no validation errors.
        return self._config is not None and not self.validation_errors

    def get_secure_value(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieve sensitive configuration values, prioritizing environment variables.

        Converts dot notation key to uppercase underscore notation for env var lookup.
        Example: 'api.kraken.key' becomes 'API_KRAKEN_KEY'.
        Logs source (env or config) but not the value itself unless default is returned.
        """
        env_var_name = key.replace(".", "_").upper()
        env_value = os.environ.get(env_var_name)

        if env_value is not None:
            # Log that it was found in env, but not the value
            self._logger.info(
                ("Retrieved secure value for '%s' "
                 "from environment variable '%s'."),
                key,
                env_var_name,
            )
            return env_value
        # Fall back to config file using the regular 'get' method
        config_value = self.get(key, default)  # 'get' handles logging for missing keys
        if config_value is not None and config_value != default:
            # Log that it was found in config, but not the value
            self._logger.debug(
                ("Retrieved secure value for '%s' from config file "
                 "(env var '%s' not set)."),
                key,
                env_var_name,
            )
        elif config_value is None and default is None:
            # Log warning only if it's truly missing (not just using default=None)
            self._logger.warning(
                ("Secure value for '%s' not found in environment or config file. "
                 "Returning None."),
                key,
            )

        # Ensure we return an Optional[str] to satisfy mypy
        if config_value is None:
            return None
        if isinstance(config_value, str):
            return config_value
        # Convert non-string values to strings
        self._logger.debug("Converting non-string config value for '%s' to string.", key)
        return str(config_value)

    def get_secure_api_key(self, service_name: str, key_name: str = "key") -> Optional[str]:
        """Retrieve a specific API key securely for a given service."""
        full_key = f"api.{service_name}.{key_name}"
        return self.get_secure_value(full_key)

    def get_secure_api_secret(
        self, service_name: str, secret_name: str = "secret"
    ) -> Optional[str]:
        """Retrieve a specific API secret securely for a given service."""
        full_key = f"api.{service_name}.{secret_name}"
        return self.get_secure_value(full_key)
