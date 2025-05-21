"""Provide configuration management for the Gal-Friday trading system.

This module handles loading, validating, and accessing application configuration from
YAML files. It provides secure access to sensitive information like API keys and supports
both file-based configuration and environment variable overrides.
It also includes functionality to watch for configuration file changes and publish
update events.
"""

# Configuration Manager Module

import asyncio  # For event publishing
from decimal import Decimal
from functools import reduce
import logging
import operator
import os
from pathlib import Path  # Added for PTH110 and PTH123
from typing import TYPE_CHECKING, Any, Optional

import yaml

# Import FileSystemEvent conditionally for type checking
if TYPE_CHECKING:
    from watchdog.events import FileSystemEvent

# Watchdog for file monitoring - ensure this is in requirements.txt
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    Observer = None
    FileSystemEventHandler = None
    logging.getLogger(__name__).warning(
        "'watchdog' library not found. Dynamic config reloading will be disabled. "
        "Please install it: pip install watchdog"
    )

# Conditional import for PubSubManager and Event types
if TYPE_CHECKING:
    from .core.pubsub import PubSubManager

log = logging.getLogger(__name__)


class ConfigChangeHandler(FileSystemEventHandler if FileSystemEventHandler else object):
    """Handles file system events for configuration file changes."""

    def __init__(self, config_manager: "ConfigManager", config_file_path: Path) -> None:
        super().__init__()
        self.config_manager = config_manager
        self.config_file_path = config_file_path
        self.logger = logging.getLogger(f"{__name__}.ConfigChangeHandler")
        self._last_triggered_time = 0.0  # For simple debouncing
        self._debounce_period = 2.0  # Seconds
        self._pending_tasks: set[asyncio.Task[None]] = set()  # Track pending tasks

    def on_modified(self, event: "FileSystemEvent") -> None:
        """Handle file or directory modification events.

        Args:
            event: The file system event that triggered this callback.
        """
        if not event.is_directory and Path(event.src_path) == self.config_file_path:
            current_time = asyncio.get_event_loop().time()
            if (current_time - self._last_triggered_time) < self._debounce_period:
                self.logger.debug(
                    "Debouncing config file modification event for %s",
                    event.src_path
                )
                return
            self._last_triggered_time = current_time

            self.logger.info(
                "Configuration file %s modified. Scheduling reload.",
                event.src_path
            )
            # Use the loop provided to ConfigManager to schedule the task
            loop = getattr(self.config_manager, "_loop", None)
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.config_manager.handle_config_file_change(), loop
                )
            else:
                # Fallback if loop isn't available for threadsafe scheduling.
                # This might happen if watchdog events come from a non-async thread.
                # A more robust solution would use an async-compatible watchdog or a queue.
                warning_msg = (
                    "Event loop not available or not running for scheduling config "
                    "reload from watchdog thread. Attempting direct task creation "
                    "(may fail if not on main thread)."
                )
                self.logger.warning(warning_msg)
                try:
                    # Store task reference to prevent garbage collection
                    task = asyncio.create_task(
                        self.config_manager.handle_config_file_change()
                    )
                    # Store task reference if needed later
                    self._pending_tasks.add(task)
                    task.add_done_callback(self._pending_tasks.discard)
                except RuntimeError:
                    self.logger.exception(
                        "RuntimeError creating task for config reload (likely wrong thread)"
                    )


class ConfigManager:
    """Manage loading, accessing, and dynamic reloading of app configuration."""

    _EXPECTED_TRADING_PAIR_COMPONENTS = 2

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        logger_service: Optional[logging.Logger] = None,
        pubsub_manager: Optional["PubSubManager"] = None,  # Added PubSubManager
        loop: Optional[asyncio.AbstractEventLoop] = None,  # Optional asyncio loop
    ) -> None:
        """
        Initialize the ConfigManager, load config, and validate it.

        Args:
            config_path: Path to the YAML configuration file.
            logger_service: Optional logger instance.
            pubsub_manager: Optional PubSubManager for publishing config update events.
            loop: Optional asyncio event loop for scheduling tasks from watchdog.
        """
        self._config_path_str = config_path
        self._config_file_path = Path(self._config_path_str).resolve()
        self._config: Optional[dict] = None
        self.validation_errors: list[str] = []
        self._pubsub_manager = pubsub_manager
        # Ensure _loop is captured correctly, preferably from the running application context
        if loop:
            self._loop = loop
        else:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._logger.info("No running event loop, created a new one for ConfigManager.")

        self._observer: Optional[Observer] = None
        self._config_reload_lock = asyncio.Lock()  # Lock for reloading
        # Store the previous prediction_service config for comparison
        self._previous_prediction_service_config: Optional[dict[str, Any]] = None

        if Observer and FileSystemEventHandler:
            self._event_handler = ConfigChangeHandler(self, self._config_file_path)
        else:
            self._event_handler = None  # Watchdog not available

        self._logger = logger_service or logging.getLogger(__name__)
        self._logger.info(
            "Initializing ConfigManager with path: %s", self._config_file_path
        )

        self.load_config()
        self.validation_errors = self.validate_configuration()
        if not self.is_valid():
            self._logger.error("Initial configuration is invalid. Please check errors above.")
            # Potentially raise an error here if initial valid config is mandatory

    async def handle_config_file_change(self) -> None:  # noqa: PLR0912
        """
        Handle configuration file changes.

        This method is called by the Watchdog handler when the configuration file
        changes. It reloads the configuration and publishes an update event if
        meaningful changes are detected.
        """
        if self._config_reload_lock.locked():
            self._logger.info(
                "Configuration reload already in progress. Skipping duplicate trigger."
            )
            return

        async with self._config_reload_lock:
            self._logger.info(
                "Handling detected change for %s", self._config_file_path
            )

            # Store current prediction_service config before reloading, if it exists and is valid
            # This assumes self._config is from the *previous* valid load.
            if self._config and isinstance(self._config.get("prediction_service"), dict):
                self._previous_prediction_service_config = self._config.get(
                    "prediction_service"
                ).copy()
            else:
                self._previous_prediction_service_config = None

            reloaded_successfully = True
            try:
                self.load_config()  # This updates self._config
                new_validation_errors = self.validate_configuration()
                self.validation_errors = new_validation_errors

                if not self.is_valid():
                    self._logger.error(
                        "Reloaded configuration is invalid. Not publishing update event."
                    )
                    reloaded_successfully = False
                else:
                    self._logger.info(
                        "Configuration reloaded and validated successfully after file change."
                    )

            except Exception:
                self._logger.exception("Error during automated config reload process.")
                reloaded_successfully = False

            if reloaded_successfully and self._pubsub_manager and self._config is not None:
                from .core.events import PredictionConfigUpdatedEvent

                current_prediction_service_config = self._config.get("prediction_service")

                # Check for meaningful changes before publishing
                if self._has_prediction_config_changed(current_prediction_service_config):
                    if current_prediction_service_config is not None and isinstance(
                        current_prediction_service_config, dict
                    ):
                        self._logger.info(
                            "Meaningful changes detected in 'prediction_service' config. "
                            "Publishing PREDICTION_CONFIG_UPDATED event."
                        )
                        try:
                            event = PredictionConfigUpdatedEvent.create(
                                source_module=self.__class__.__name__,
                                new_config=current_prediction_service_config,
                            )
                            await self._pubsub_manager.publish(event)
                            # Update the baseline for next comparison
                            self._previous_prediction_service_config = (
                                current_prediction_service_config.copy()
                            )
                        except Exception as e_pub:
                            self._logger.exception(
                                "Failed to publish PREDICTION_CONFIG_UPDATED event.",
                                exc_info=e_pub,
                            )
                    else:
                        self._logger.error(
                            "'prediction_service' section not found or not a dict in "
                            "reloaded config after change detection. Cannot publish update."
                        )
                else:
                    self._logger.info(
                        "No meaningful changes detected in 'prediction_service' "
                        "config. Update event not published."
                    )
            elif reloaded_successfully and self._pubsub_manager is None:
                self._logger.warning(
                    "Config reloaded but PubSubManager not available to publish update event."
                )

    def _has_prediction_config_changed(self, new_config_section: Optional[dict[str, Any]]) -> bool:
        """Compare the new prediction_service config section with the previous one.

        Returns
        -------
            bool: True if the config has changed, False otherwise.
        """
        if self._previous_prediction_service_config is None and new_config_section is None:
            return False  # Both are None, no change
        if self._previous_prediction_service_config is None or new_config_section is None:
            return True  # One is None, the other is not, definitely a change

        # Compare key fields. Add more fields if their change should trigger a reload.
        # For simplicity, comparing the entire dicts is robust if they are simple structures.
        # Deep comparison might be needed for nested mutable structures if not copied properly.
        # Assuming models and ensemble_strategy are the main drivers for PredictionService reload.

        prev_models = self._previous_prediction_service_config.get("models")
        new_models = new_config_section.get("models")
        prev_strategy = self._previous_prediction_service_config.get("ensemble_strategy")
        new_strategy = new_config_section.get("ensemble_strategy")
        # Add other critical keys like ensemble_weights if they should trigger reload on change.


        # Check if models or strategy have changed
        return bool(
            prev_models != new_models
            or prev_strategy != new_strategy
            # Add more conditions here if needed
        )

    def start_watching(self) -> None:
        """Start watching the configuration file for changes if watchdog is available."""
        if (
            not self._observer and Observer and self._event_handler
        ):  # Check if already started or if watchdog available
            self._observer = Observer()
            try:
                # Watch the directory containing the file, as watching single files can be
                # problematic on some OS.
                watch_path = str(self._config_file_path.parent)
                self._observer.schedule(
                    self._event_handler, watch_path, recursive=False
                )
                self._observer.start()
                self._logger.info(
                    "Started watching configuration file directory: %s (for changes to %s)",
                    watch_path,
                    self._config_file_path.name,
                )
            except Exception as e:
                self._logger.exception(
                    "Failed to start configuration file observer for %s",
                    self._config_file_path,
                    exc_info=e,
                )
                self._observer = None  # Reset if failed to start
        elif self._observer:
            self._logger.info("Configuration file observer already running.")
        else:
            self._logger.warning(
                "Watchdog library not available. Cannot start config file watching."
            )

    def stop_watching(self) -> None:
        """Stop watching the configuration file."""
        if self._observer:
            try:
                if self._observer.is_alive():
                    self._observer.stop()
                    self._observer.join(timeout=5)  # Wait for observer thread to finish
                self._logger.info(
                    "Stopped watching configuration file: %s", self._config_file_path
                )
            except Exception as e:
                self._logger.exception("Error stopping configuration file observer.", exc_info=e)
            finally:
                self._observer = None  # Ensure it's reset
        else:
            self._logger.info("Configuration file observer was not running.")

    def load_config(self) -> None:
        """Load or reload the configuration from the specified YAML file."""
        self._logger.info(
            "Attempting to load configuration from: %s", self._config_file_path
        )
        try:
            if not self._config_file_path.exists():
                self._logger.error(
                    "Configuration file not found at: %s", self._config_file_path
                )
                self._config = {}  # Set to empty dict if file not found
                return

            with self._config_file_path.open("r") as f:
                self._config = yaml.safe_load(f)
            self._logger.info(
                "Successfully loaded configuration from %s", self._config_file_path
            )
        except yaml.YAMLError as e:
            self._logger.exception(
                "Error parsing YAML configuration file: %s", self._config_file_path, exc_info=e
            )
            self._config = {}
        except OSError as e:
            self._logger.exception(
                "Error reading configuration file: %s", self._config_file_path, exc_info=e
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
            "Could not interpret value for key '%s' ('%s') as bool. Returning default %s.",
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
            "Value for key '%s' is not a list (type: %s). Returning default %s.",
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
            "Value for key '%s' is not a dict (type: %s). Returning default %s.",
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
            "key": self.get_secure_api_key(service_name),
            "secret": self.get_secure_api_secret(service_name),
            # Add other potential credential fields if needed, e.g., 'password'
            # 'password': self.get_secure_value(f'api.{service_name}.password')
        }

    def reload_config(self) -> list[str]:
        """Manually reload the configuration from the file and re-validate it.

        This method is for explicit reload calls. Automated reloading is handled by watchdog.
        Updates `self.validation_errors` with the results of the new validation.

        Returns
        -------
            List of validation errors encountered during the reload and validation process.
        """
        self._logger.info(
            "Manual reload_config called for: %s", self._config_file_path
        )
        # Simply call load and validate. Event publishing is handled by
        # handle_config_file_change if this manual reload was triggered by an
        # external system that mimics a file change. If this is purely an
        # internal programmatic reload without a file change event, publishing
        # an event here would require careful thought about its implications.
        self.load_config()
        self.validation_errors = self.validate_configuration()
        if not self.validation_errors:
            self._logger.info("Configuration reloaded and validated successfully (manual call).")
        else:
            self._logger.warning(
                "Configuration reload (manual call) completed, but validation failed."
            )
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
                ("Retrieved secure value for '%s' from environment variable '%s'."),
                key,
                env_var_name,
            )
            return env_value
        # Fall back to config file using the regular 'get' method
        config_value = self.get(key, default)  # 'get' handles logging for missing keys
        if config_value is not None and config_value != default:
            # Log that it was found in config, but not the value
            self._logger.debug(
                ("Retrieved secure value for '%s' from config file (env var '%s' not set)."),
                key,
                env_var_name,
            )
        elif config_value is None and default is None:
            # Log warning only if it's truly missing (not just using default=None)
            self._logger.warning(
                ("Secure value for '%s' not found in environment or config file. Returning None."),
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

    def get_secure_api_key(self, service_name: str) -> Optional[str]:
        """Retrieve a specific API key securely for a given service."""
        full_key = f"api.{service_name}.key"
        return self.get_secure_value(full_key)

    def get_secure_api_secret(self, service_name: str) -> Optional[str]:
        """Retrieve a specific API secret securely for a given service."""
        full_key = f"api.{service_name}.secret"
        return self.get_secure_value(full_key)
