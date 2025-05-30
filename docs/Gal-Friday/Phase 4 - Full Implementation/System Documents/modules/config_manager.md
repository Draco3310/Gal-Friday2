# ConfigManager Module Documentation

## Module Overview

The `config_manager.py` module is responsible for loading, validating, and providing access to the application's configuration settings. It ensures that the application has a reliable and consistent way to retrieve configuration parameters, including sensitive data like API keys, while also supporting dynamic reloading and validation.

## Key Features

-   **YAML File-Based Configuration:** Configuration is primarily managed through a YAML file, allowing for a human-readable and structured format.
-   **Environment Variable Overrides:** Supports overriding configuration values with environment variables, particularly for sensitive data, enhancing security and deployment flexibility.
-   **Secure Handling of Sensitive Data:** Provides dedicated methods to retrieve API keys and other sensitive information, prioritizing environment variables over file-based configuration for these values.
-   **Explicit Configuration Reload Mechanism:** Allows the configuration to be reloaded at runtime without restarting the application. Includes safeguards to prevent loading invalid configurations.
-   **Data Type Casting:** Offers convenient getter methods that automatically cast configuration values to common data types such as `int`, `float`, `Decimal`, `bool`, `list`, and `dict`, with error handling for invalid casts.
-   **Built-in Configuration Validation:** Includes a validation system to check the integrity and correctness of the loaded configuration against predefined rules and schemas.

## Class `ConfigManager`

The `ConfigManager` class is the core component of this module.

### Initialization (`__init__`)

-   **Parameters:**
    -   `config_path (str)`: The file path to the main YAML configuration file.
    -   `logger_service (logging.Logger)`: An instance of a logger for logging messages and errors.
-   **Actions:**
    -   Initializes the `config_path` and `logger_service`.
    -   Sets up internal state variables for storing the configuration, validation errors, and a lock for thread-safe operations.
    -   Performs an initial load of the configuration by calling `load_config()`.
    -   Validates the loaded configuration using `validate_configuration()`.

### Core Methods

-   **`load_config()`**:
    -   Loads or reloads the configuration from the YAML file specified by `config_path`.
    -   Handles potential `FileNotFoundError` if the configuration file does not exist.
    -   Handles `yaml.YAMLError` if there are issues parsing the YAML file.
    -   Logs success or failure of the loading process.

-   **`get(key: str, default: Any = None) -> Any`**:
    -   Retrieves a configuration value using a dot-separated key (e.g., `"database.host"`).
    -   If the key is not found, it returns the provided `default` value.
    -   Navigates nested dictionaries based on the dot-separated key.

-   **Type-Specific Getters:**
    -   `get_int(key: str, default: int = None) -> Optional[int]`
    -   `get_float(key: str, default: float = None) -> Optional[float]`
    -   `get_decimal(key: str, default: Decimal = None) -> Optional[Decimal]`
    -   `get_bool(key: str, default: bool = None) -> Optional[bool]`
    -   `get_list(key: str, default: list = None) -> Optional[list]`
    -   `get_dict(key: str, default: dict = None) -> Optional[dict]`
    -   These methods retrieve a value for the given key and attempt to cast it to the specified type.
    -   They accept a `default` value to return if the key is not found or if casting fails.
    -   Log errors if casting is unsuccessful.

-   **`validate_configuration() -> List[str]`**:
    -   Validates the currently loaded configuration against a predefined schema and rules.
    -   Checks for the presence and correct types of essential configuration sections and parameters, such as `trading`, `risk`, and `api`.
    -   Populates the `validation_errors` property with any issues found.
    -   Returns a list of error messages. An empty list indicates a valid configuration.

-   **`_is_valid_trading_pair(pair: str) -> bool`**:
    -   A private helper method to validate the format of a trading pair string (e.g., "BTC/USD").
    -   Typically checks for uppercase letters, a slash separator, and minimum/maximum length.

-   **Private Validation Helpers:**
    -   `_validate_trading_section(errors: List[str])`: Validates the `trading` section of the configuration (e.g., checks `trading_pairs`, `default_exchange`).
    -   `_validate_risk_section(errors: List[str])`: Validates the `risk` section (e.g., `max_total_drawdown_pct`, `max_risk_per_trade_pct`).
    -   `_validate_api_section(errors: List[str])`: Validates the `api` section, ensuring necessary API key placeholders or structures are present.

-   **`get_trading_pairs() -> List[str]`**:
    -   Returns a list of configured trading pairs from the `trading.trading_pairs` section of the configuration.
    -   Returns an empty list if not found or invalid.

-   **`get_risk_parameters() -> Dict`**:
    -   Returns a dictionary containing risk management parameters from the `risk` section.

-   **`get_strategy_parameters(strategy_id: str) -> Optional[Dict]`**:
    -   Returns parameters for a specific trading strategy identified by `strategy_id`.
    -   Looks for strategy configurations typically under a `strategies.<strategy_id>` path in the YAML.

-   **`get_api_keys(service_name: str) -> Optional[Dict[str, str]]`**:
    -   Retrieves the API key and secret for a specified service (e.g., "kraken", "binance").
    -   It first checks the configuration file under `api.<service_name>`.
    -   Crucially, it then attempts to override these values with environment variables for enhanced security. For a service named `kraken`, it would look for `KRAKEN_API_KEY` and `KRAKEN_API_SECRET`.
    -   Returns a dictionary like `{"api_key": "key_value", "api_secret": "secret_value"}` or `None` if not found.

-   **`reload_config()`**:
    -   Explicitly triggers a reload of the configuration from the YAML file.
    -   It stores the current valid configuration before attempting to load the new one.
    -   If the reload fails (e.g., file not found, parsing error) or if the newly loaded configuration is invalid (checked via `validate_configuration()`), it reverts to the previously loaded valid configuration.
    -   This ensures the application continues to run with a known good configuration if the new one is problematic.
    -   Logs the outcome of the reload attempt.

-   **`is_valid() -> bool`**:
    -   Returns `True` if the current configuration has no validation errors (i.e., `validation_errors` list is empty), `False` otherwise.

-   **`get_secure_value(key: str, default: Any = None) -> Optional[str]`**:
    -   A generic method to retrieve sensitive configuration values.
    -   It first attempts to get the value from the configuration file using the provided `key`.
    -   Then, it constructs an environment variable name based on the `key` (e.g., `database.admin_password` might become `DATABASE_ADMIN_PASSWORD`).
    -   If the environment variable is set, its value takes precedence and is returned. Otherwise, the value from the config file (or default) is used.
    -   This method is the foundation for `get_secure_api_key` and `get_secure_api_secret`.

-   **`get_secure_api_key(service_name: str) -> Optional[str]`**:
    -   A specialized getter for retrieving an API key for a given `service_name`.
    -   Uses `get_secure_value` internally, checking `api.<service_name>.api_key` in the config file and then the corresponding environment variable (e.g., `SERVICE_NAME_API_KEY`).

-   **`get_secure_api_secret(service_name: str) -> Optional[str]`**:
    -   A specialized getter for retrieving an API secret for a given `service_name`.
    -   Uses `get_secure_value` internally, checking `api.<service_name>.api_secret` in the config file and then the corresponding environment variable (e.g., `SERVICE_NAME_API_SECRET`).

### Properties

-   **`validation_errors: List[str]`**:
    -   A read-only property that returns a list of strings, where each string is a description of a validation error found in the current configuration.
    -   An empty list signifies that the configuration is valid.

## Configuration File Structure (Example)

The `config_manager` expects a YAML file with a structure similar to the following:

```yaml
# Main application settings
application:
  name: "GalFriday AlgoTrader"
  version: "1.0.0"
  environment: "development" # or "production"

# Database configuration
database:
  type: "postgresql"
  host: "localhost"
  port: 5432
  username: "admin"
  # password: "secure_password" # Recommended to set via environment variable: DATABASE_PASSWORD

# Trading settings
trading:
  default_exchange: "kraken"
  trading_pairs:
    - "BTC/USD"
    - "ETH/USD"
    - "ADA/USD"
  default_leverage: 1.0
  # Strategy-specific parameters
  strategies:
    rsi_momentum:
      rsi_period: 14
      buy_threshold: 30
      sell_threshold: 70
    mean_reversion:
      window_size: 20
      std_dev_multiplier: 2.0

# Risk management parameters
risk:
  max_total_drawdown_pct: 15.0
  max_risk_per_trade_pct: 1.0
  max_concurrent_trades: 5
  stop_loss_pct: 2.0 # Default stop-loss if not set by strategy

# API keys and service configurations
# It's highly recommended to set actual API keys via environment variables
# e.g., KRAKEN_API_KEY, KRAKEN_API_SECRET
api:
  kraken:
    api_key: "YOUR_KRAKEN_API_KEY_PLACEHOLDER" # Override with KRAKEN_API_KEY env var
    api_secret: "YOUR_KRAKEN_API_SECRET_PLACEHOLDER" # Override with KRAKEN_API_SECRET env var
    # Other Kraken-specific settings
    timeout_ms: 5000
  binance:
    api_key: "YOUR_BINANCE_API_KEY_PLACEHOLDER" # Override with BINANCE_API_KEY env var
    api_secret: "YOUR_BINANCE_API_SECRET_PLACEHOLDER" # Override with BINANCE_API_SECRET env var
    # Other Binance-specific settings
    recv_window: 5000

# Logging configuration
logging:
  level: "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_file: "galfriday.log"
  log_to_console: true

# Other service configurations (e.g., notification services)
services:
  telegram:
    bot_token: "YOUR_TELEGRAM_BOT_TOKEN_PLACEHOLDER" # Override with TELEGRAM_BOT_TOKEN env var
    chat_id: "YOUR_TELEGRAM_CHAT_ID"
```

## Error Handling

The `config_manager` module incorporates robust error handling:

-   **Missing Configuration File:** If the specified `config_path` does not point to an existing file, a `FileNotFoundError` is caught during `load_config()`, an error is logged, and the internal configuration remains empty or unchanged (if reloading).
-   **YAML Parsing Errors:** If the configuration file is malformed and cannot be parsed as valid YAML, a `yaml.YAMLError` is caught during `load_config()`. An error is logged, and the configuration is not loaded or updated.
-   **Validation Errors:** After a configuration is loaded (or reloaded), `validate_configuration()` is called. Any discrepancies against the predefined schema or rules are collected in the `validation_errors` list. The `is_valid()` method can be used to check this status. The application can then decide how to handle an invalid configuration (e.g., log errors and exit, or operate with default/limited functionality).
-   **Type Casting Errors:** When using type-specific getters (e.g., `get_int()`, `get_decimal()`), if a value cannot be converted to the target type, an error is logged, and the method returns the provided `default` value (or `None`).
-   **Key Not Found:** If a requested configuration key is not found using `get()` or its typed variants, the provided `default` value is returned. No error is logged for this case by default, as it's considered a normal operational scenario.

## Dependencies

The `config_manager.py` module relies on the following Python libraries:

-   **`yaml` (`PyYAML`):** For parsing YAML configuration files.
-   **`logging`:** For application-wide logging of events, errors, and debug information.
-   **`os`:** Used for accessing environment variables (e.g., for overriding configuration values).
-   **`pathlib`:** Used for path manipulations, providing an object-oriented way to handle file system paths.
-   **`decimal`:** Used for precise decimal arithmetic, particularly important for financial calculations (e.g., `get_decimal()`).

## Usage Example

```python
from gal_friday.config_manager import ConfigManager # Assuming ConfigManager is in this path
from decimal import Decimal
import logging
import os

# Initialize logger (or use an existing one from a central logging service)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Example: Set an environment variable for a secure API key before initializing ConfigManager
os.environ["KRAKEN_API_KEY"] = "actual_api_key_from_env"
os.environ["KRAKEN_API_SECRET"] = "actual_api_secret_from_env"

# Define the path to your configuration file
# This could be relative to your project root or an absolute path
CONFIG_FILE_PATH = "config/config.yaml" # Adjust path as needed

# Create a dummy config.yaml for the example to run
# In a real application, this file would already exist.
os.makedirs("config", exist_ok=True)
with open(CONFIG_FILE_PATH, "w") as f:
    f.write("""
application:
  name: "GalFriday AlgoTrader"
  version: "1.0.0"

database:
  host: "db.example.com"
  port: 5432

trading:
  trading_pairs:
    - "BTC/USD"
    - "ETH/EUR"
  default_exchange: "kraken"

risk:
  max_total_drawdown_pct: 10.0 # This will be read as a Decimal

api:
  kraken:
    # api_key and api_secret will be overridden by environment variables
    api_key: "placeholder_key_in_file"
    api_secret: "placeholder_secret_in_file"
  other_service:
    token: "service_token_in_file" # Example of a secure value not overridden by env
""")

# Initialize ConfigManager
config_manager = ConfigManager(config_path=CONFIG_FILE_PATH, logger_service=logger)

if not config_manager.is_valid():
    logger.error("Configuration is invalid. Halting application.")
    logger.error("Validation Errors: %s", config_manager.validation_errors)
    # In a real application, you might exit or take other corrective actions
    # For this example, we'll just print the errors and continue to show other features.
else:
    logger.info("Configuration loaded and validated successfully.")

# Get specific values
app_name = config_manager.get("application.name", "DefaultAppName")
logger.info(f"Application Name: {app_name}")

db_host = config_manager.get("database.host", "localhost")
logger.info(f"Database Host: {db_host}")

# Example of getting a typed value (Decimal)
max_drawdown = config_manager.get_decimal("risk.max_total_drawdown_pct", Decimal("15.0"))
logger.info(f"Max Total Drawdown: {max_drawdown}% (Type: {type(max_drawdown)})")

# Get a boolean value (assuming it exists, otherwise default)
feature_flag = config_manager.get_bool("features.new_feature_enabled", False)
logger.info(f"New Feature Enabled: {feature_flag}")

# Get a list
trading_pairs = config_manager.get_trading_pairs() # Uses specialized getter
logger.info(f"Trading Pairs: {trading_pairs}")

# Retrieve secure API keys for Kraken
# These should come from environment variables set earlier
kraken_api_key = config_manager.get_secure_api_key("kraken")
kraken_api_secret = config_manager.get_secure_api_secret("kraken")
logger.info(f"Kraken API Key (from env): {kraken_api_key}")
logger.info(f"Kraken API Secret (from env): {kraken_api_secret}")

# Retrieve a secure value that might only be in the file (if not set in env)
# For this, we'll use the generic get_secure_value
# Let's assume OTHER_SERVICE_TOKEN is NOT set as an environment variable
other_service_token = config_manager.get_secure_value("api.other_service.token")
logger.info(f"Other Service Token (from file): {other_service_token}")

# Example of a missing value with a default
non_existent_value = config_manager.get("non_existent.path", "default_value_for_missing")
logger.info(f"Non-existent value: {non_existent_value}")

# --- Example of reloading configuration ---
# logger.info("Simulating configuration file change and reloading...")
# # Modify the config file (in a real scenario, this would be an external change)
# with open(CONFIG_FILE_PATH, "w") as f:
#     f.write("""
# application:
#   name: "GalFriday AlgoTrader v2" # Changed value
# database:
#   host: "new_db.example.com"
# trading:
#   trading_pairs: ["XRP/USD"]
# risk:
#   max_total_drawdown_pct: 12.5
# api:
#   kraken: {}
# """)
#
# if config_manager.reload_config():
#     logger.info("Configuration reloaded successfully.")
#     new_app_name = config_manager.get("application.name")
#     logger.info(f"New Application Name: {new_app_name}")
#     new_trading_pairs = config_manager.get_trading_pairs()
#     logger.info(f"New Trading Pairs: {new_trading_pairs}")
# else:
#     logger.error("Failed to reload configuration or reloaded configuration was invalid.")
#     logger.info(f"Still using previous app name: {config_manager.get('application.name')}")

# Clean up the dummy config file and directory
# os.remove(CONFIG_FILE_PATH)
# os.rmdir("config")
```

## Adherence to Standards

This documentation aims to align with best practices for software documentation, drawing inspiration from principles found in standards such as:

-   **ISO/IEC/IEEE 26512:2018** (Acquirers and suppliers of information for users)
-   **ISO/IEC/IEEE 12207** (Software life cycle processes)
-   **ISO/IEC/IEEE 15288** (System life cycle processes)

While not a formal certification, the structure and content strive for clarity, completeness, and accuracy to support developers and maintainers of the `config_manager` module.
