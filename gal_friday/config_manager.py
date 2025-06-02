"""Provide configuration management for the Gal-Friday trading system.

This module handles loading, validating, and accessing application configuration from
YAML files. It provides secure access to sensitive information like API keys and supports
both file-based configuration and environment variable overrides.

Configuration changes require explicit reload or system restart for safety during live trading.
"""

# Configuration Manager Module

import logging
import operator
import os
from decimal import Decimal
from functools import reduce
from pathlib import Path
from typing import Any

import yaml

# Conditional import for PubSubManager type checking only

log = logging.getLogger(__name__)


class ConfigManager:
    """Manage loading, accessing, and explicit reloading of app configuration."""

    _EXPECTED_TRADING_PAIR_COMPONENTS = 2
    _MAX_RISK_PERCENTAGE = 100.0

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        logger_service: logging.Logger | None = None,
    ) -> None:
        """Initialize the ConfigManager and load configuration.

        Args:
            config_path: Path to the YAML configuration file.
            logger_service: Optional logger instance.
        """
        self._config_path_str = config_path
        self._config_file_path = Path(self._config_path_str).resolve()
        self._config: dict | None = None
        self.validation_errors: list[str] = []

        # Ensure logger is always initialized
        self._logger: logging.Logger = logger_service or logging.getLogger(__name__)
        self._logger.info(
            "Initializing ConfigManager with path: %s",
            self._config_file_path,
        )

        self.load_config()
        self.validation_errors = self.validate_configuration()
        if not self.is_valid():
            self._logger.error("Initial configuration is invalid. Please check errors above.")

    def load_config(self) -> None:
        """Load or reload the configuration from the specified YAML file."""
        self._logger.info(
            "Attempting to load configuration from: %s",
            self._config_file_path,
        )
        try:
            if not self._config_file_path.exists():
                self._logger.error(
                    "Configuration file not found at: %s",
                    self._config_file_path,
                )
                self._config = {}  # Set to empty dict if file not found
                return

            with self._config_file_path.open("r") as f:
                self._config = yaml.safe_load(f)
            self._logger.info(
                "Successfully loaded configuration from %s",
                self._config_file_path,
            )
        except yaml.YAMLError as e:
            self._logger.exception(
                "Error parsing YAML configuration file: %s",
                self._config_file_path,
                exc_info=e,
            )
            self._config = {}
        except OSError as e:
            self._logger.exception(
                "Error reading configuration file: %s",
                self._config_file_path,
                exc_info=e,
            )
            self._config = {}
        except Exception as e:
            self._logger.exception(
                "Error loading configuration from %s",
                self._config_file_path,
                exc_info=e,
            )
            self._config = {}

        if not isinstance(self._config, dict):
            self._logger.error(
                "Configuration file %s did not load as a dictionary. "
                "Loaded type: %s. Setting config to empty dict.",
                self._config_file_path,
                type(self._config),
            )
            self._config = {}

    def get(self, key: str, default: Any | None = None) -> Any:  # noqa: ANN401
        """Retrieve a configuration value using a dot-separated key.

        Example:
            config.get('database.postgres.host', 'localhost')

        Args:
        ----
            key: The dot-separated key string.
            default: The value to return if the key is not found.

        Returns:
        -------
            The configuration value or the default.
        """
        if self._config is None:
            self._logger.warning(
                "Configuration accessed before it was loaded or after a loading error.",
            )
            return default

        try:
            return reduce(operator.getitem, key.split("."), self._config)
        except (KeyError, TypeError):
            # KeyError if a key in the path doesn't exist
            # TypeError if trying to index into a non-dictionary
            self._logger.debug(
                "Key '%s' not found in configuration. Returning default: %s",
                key,
                default,
            )
            return default
        except Exception as e:
            self._logger.exception(
                "Unexpected error retrieving key '%s' from configuration.",
                key,
                exc_info=e,
            )
            return default

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
            except (ValueError, TypeError): # BLE001
                self._logger.warning(
                    "Invalid default value '%s' for get_decimal, using 0.0",
                    default,
                )
                default = Decimal("0.0")

        value = self.get(key, default)
        try:
            # Convert to string first to handle floats/ints correctly
            return Decimal(str(value))
        except (ValueError, TypeError) as e: # BLE001
            self._logger.warning(
                "Could not convert value for key '%s' ('%s') to Decimal. "
                "Returning default %s. Error: %s",
                key,
                value,
                default,
                e,
            )
            return default

    def get_bool(self, key: str, *, default: bool = False) -> bool: # FBT001, FBT002
        """Retrieve a config value and attempt to cast it to a boolean."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        try:
            return bool(value)
        except (ValueError, TypeError) as e:
            self._logger.warning(
                "Could not convert value for key '%s' ('%s') to bool. "
                "Returning default %s. Error: %s",
                key,
                value,
                default,
                e,
            )
            return default

    def get_list(self, key: str, default: list[Any] | None = None) -> list[Any]:
        """Retrieve a config value and ensure it's a list."""
        if default is None:
            default = []
        value = self.get(key, default)
        if isinstance(value, list):
            return value
        return default

    def get_dict(self, key: str, default: dict | None = None) -> dict:
        """Retrieve a config value and ensure it's a dictionary."""
        if default is None:
            default = {}
        value = self.get(key, default)
        if isinstance(value, dict):
            return value
        return default

    def validate_configuration(self) -> list[str]:
        """Validate the loaded configuration and return a list of errors."""
        errors = []
        if self._config is None or not isinstance(self._config, dict):
            errors.append("Configuration could not be loaded or is not a valid dictionary")
            return errors

        self._validate_trading_section(errors)
        self._validate_risk_section(errors)
        self._validate_api_section(errors)

        return errors

    def _is_valid_trading_pair(self, pair: str) -> bool:
        """Check if a trading pair string is in valid format (e.g., 'XRP/USD')."""
        if not isinstance(pair, str):
            return False
        parts = pair.split("/")
        return (
            len(parts) == self._EXPECTED_TRADING_PAIR_COMPONENTS
            and all(len(part.strip()) > 0 for part in parts)
        )

    def _validate_trading_section(self, errors: list[str]) -> None:
        """Validate the trading configuration section."""
        trading_config = self.get("trading", {})
        if not isinstance(trading_config, dict):
            errors.append("'trading' section must be a dictionary")
            return

        # Validate trading pairs
        pairs = trading_config.get("pairs", [])
        if not isinstance(pairs, list) or len(pairs) == 0:
            errors.append("'trading.pairs' must be a non-empty list")
        else:
            invalid_pair_errors = [
                f"Invalid trading pair format: '{pair}' (expected format: 'BASE/QUOTE')"
                for pair in pairs
                if not self._is_valid_trading_pair(pair)
            ]
            if invalid_pair_errors:
                errors.extend(invalid_pair_errors) # PERF401

        # Validate exchange
        exchange = trading_config.get("exchange")
        if not isinstance(exchange, str) or len(exchange.strip()) == 0:
            errors.append("'trading.exchange' must be a non-empty string")

    def _validate_risk_section(self, errors: list[str]) -> None:
        """Validate the risk management configuration section."""
        risk_config = self.get("risk", {})
        if not isinstance(risk_config, dict):
            errors.append("'risk' section must be a dictionary")
            return

        # Validate key risk parameters
        # E501
        required_fields = [
            "max_total_drawdown_pct",
            "max_daily_drawdown_pct",
            "risk_per_trade_pct",
        ]
        for field in required_fields:
            value = risk_config.get(field)
            if value is None:
                errors.append(f"'risk.{field}' is required")
            else:
                try:
                    float_val = float(value)
                    # PLR2004: Using Decimal for comparison if values are financial
                    if float_val <= 0 or float_val >= self._MAX_RISK_PERCENTAGE:
                        errors.append(f"'risk.{field}' must be between 0 and 100 (exclusive)")
                except (ValueError, TypeError):
                    errors.append(f"'risk.{field}' must be a valid number")

    def _validate_api_section(self, errors: list[str]) -> None:
        """Validate the API configuration section."""
        api_config = self.get("api", {})
        if not isinstance(api_config, dict):
            errors.append("'api' section must be a dictionary")
            return

        # Check for exchange-specific API configuration
        exchange = self.get("trading.exchange", "")
        if exchange:
            exchange_api_config = api_config.get(exchange.lower(), {})
            if not isinstance(exchange_api_config, dict):
                errors.append(f"'api.{exchange.lower()}' section must be a dictionary")
            else:
                # Check for required API fields (keys may be in environment variables)
                required_api_fields = ["api_key", "api_secret"]
                # PERF401: Use list comprehension
                missing_fields = [ # E501 will be fixed by reformatting comprehension
                    field
                    for field in required_api_fields
                    if not exchange_api_config.get(field)
                    and not os.getenv(f"{exchange.upper()}_{field.upper()}")
                ]
                if missing_fields:
                    errors.append(
                        f"Missing {exchange} API configuration. "
                        f"Required fields not found in config or environment: {missing_fields}",
                    )

    def get_trading_pairs(self) -> list[str]:
        """Get the list of configured trading pairs."""
        return self.get_list("trading.pairs", [])

    def get_risk_parameters(self) -> dict[str, Any]:
        """Get the risk management parameters."""
        return self.get_dict("risk", {})

    def get_strategy_parameters(self, strategy_id: str) -> dict[str, Any]:
        """Get parameters for a specific strategy."""
        return self.get_dict(f"strategies.{strategy_id}", {})

    def get_api_keys(self, service_name: str) -> dict[str, str | None]:
        """Get API keys for a service, checking both config and environment."""
        service_config = self.get_dict(f"api.{service_name.lower()}", {})

        # Check environment variables as fallback/override
        api_key = service_config.get("api_key") or \
                  os.getenv(f"{service_name.upper()}_API_KEY") # E501
        api_secret = service_config.get("api_secret") or \
                     os.getenv(f"{service_name.upper()}_API_SECRET") # E501

        return {
            "api_key": api_key,
            "api_secret": api_secret,
        }

    def reload_config(self) -> list[str]:
        """Explicitly reload configuration from file.

        This method provides controlled configuration reloading that requires
        explicit user action (CLI command or system restart) rather than
        automatic file watching.

        Returns:
            List of validation errors (empty if successful)
        """
        self._logger.info("Explicitly reloading configuration...")

        # Store current config as backup
        backup_config = self._config
        backup_errors = self.validation_errors

        try:
            self.load_config()
            new_validation_errors = self.validate_configuration()

            if new_validation_errors:
                # Restore backup if new config is invalid
                self._config = backup_config
                self.validation_errors = backup_errors
                self._logger.error(
                    "Configuration reload failed validation. Restored previous configuration.",
                )
                return new_validation_errors
        except Exception as e:
            # Restore backup on any error
            self._config = backup_config
            self.validation_errors = backup_errors
            self._logger.exception("Configuration reload failed. Restored previous configuration.")
            return [f"Configuration reload failed: {e!s}"]
        else: # TRY300: This block executes if the try block completes with no exceptions
            self.validation_errors = new_validation_errors # Should be empty if we reach here
            self._logger.info("Configuration reloaded successfully.")
            return []

    def is_valid(self) -> bool:
        """Check if the current configuration is valid."""
        return len(self.validation_errors) == 0

    def get_secure_value(self, key: str, default: str | None = None) -> str | None:
        """Get a secure value (API key, secret, etc.) from config or environment.

        This method checks the configuration file first, then falls back to
        environment variables using a standardized naming convention.

        Args:
            key: Configuration key (e.g., 'kraken.api_key')
            default: Default value if not found

        Returns:
            The secure value or default
        """
        # Get from config first
        config_value = self.get(key, None)
        if config_value:
            return str(config_value)

        # Generate environment variable name from key
        # Convert 'kraken.api_key' to 'KRAKEN_API_KEY'
        env_key = key.replace(".", "_").upper()
        env_value = os.getenv(env_key)

        if env_value:
            return env_value

        # Check alternative naming patterns
        parts = key.split(".")
        if len(parts) >= self._EXPECTED_TRADING_PAIR_COMPONENTS: # PLR2004 (using class const)
            alt_env_key = f"{parts[0].upper()}_{parts[-1].upper()}"
            alt_env_value = os.getenv(alt_env_key)
            if alt_env_value:
                return alt_env_value

        return default

    def get_secure_api_key(self, service_name: str) -> str | None:
        """Get API key for a service."""
        return self.get_secure_value(f"{service_name.lower()}.api_key")

    def get_secure_api_secret(self, service_name: str) -> str | None:
        """Get API secret for a service."""
        return self.get_secure_value(f"{service_name.lower()}.api_secret")
