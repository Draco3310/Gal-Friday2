"""Client for accessing and retrieving feature definitions from a YAML-based Feature Registry.

This module provides a `FeatureRegistryClient` class that handles loading, parsing,
and providing access to feature definitions stored in a YAML file. It is designed
to be a reusable component for any service that needs to query the feature registry.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Configure a logger for this module
logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = "config/feature_registry.yaml"

class RegistryLoadError(Exception):
    """Custom exception raised when the feature registry fails to load or parse."""

class FeatureRegistryClient:
    """
    A client for loading and accessing feature definitions from a YAML registry file.

    This client loads feature definitions upon instantiation and provides methods
    to retrieve specific definitions, all feature keys, or parts of a definition
    (like output properties, calculator type, or parameters).

    Attributes:
        registry_path (Path): The path to the YAML feature registry file.
    """

    def __init__(self, registry_path: Path | str = DEFAULT_REGISTRY_PATH) -> None:
        """
        Initializes the FeatureRegistryClient and loads data from the registry.

        Args:
            registry_path: The path to the YAML feature registry file.
                           Defaults to `config/feature_registry.yaml`.

        Raises:
            RegistryLoadError: If the registry file cannot be found or parsed,
                               and it's considered a critical failure for the client's
                               basic operation (e.g., if the file is mandatory).
                               For now, it logs errors and sets data to None/empty.
        """
        self.registry_path: Path = Path(registry_path)
        self._registry_data: Optional[Dict[str, Any]] = None
        self._load_registry()

    def _load_registry(self) -> None:
        """
        Loads feature definitions from the YAML file specified by `self.registry_path`.

        Populates `self._registry_data` with the loaded content.
        Handles `FileNotFoundError` and `yaml.YAMLError` by logging the error
        and setting `self._registry_data` to an empty dictionary, allowing the client
        to operate in a degraded state (returning None or empty for all requests).
        A `RegistryLoadError` could be raised here if a non-functional client
        is unacceptable.
        """
        try:
            if not self.registry_path.exists():
                logger.error("Feature registry file not found: %s", self.registry_path)
                self._registry_data = {} # Operate in a degraded state
                # Optionally raise RegistryLoadError("Feature registry file not found.")
                return

            with self.registry_path.open('r') as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                logger.error(
                    "Feature registry content is not a dictionary. File: %s",
                    self.registry_path
                )
                self._registry_data = {}
                # Optionally raise RegistryLoadError("Invalid registry format: not a dictionary.")
                return

            self._registry_data = data
            logger.info(
                "Feature registry loaded successfully from %s. Found %d definitions.",
                self.registry_path, len(self._registry_data)
            )

        except yaml.YAMLError as e:
            logger.exception(
                "Error parsing YAML in feature registry %s: %s", self.registry_path, e
            )
            self._registry_data = {}
            # Optionally raise RegistryLoadError(f"YAML parsing error: {e}")
        except Exception as e: # Catch any other unexpected errors during file I/O or loading
            logger.exception(
                "Unexpected error loading feature registry %s: %s", self.registry_path, e
            )
            self._registry_data = {}
            # Optionally raise RegistryLoadError(f"Unexpected loading error: {e}")

    def get_feature_definition(self, feature_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the raw configuration dictionary for a given feature key.

        Args:
            feature_key: The unique key of the feature (e.g., "rsi_14_default").

        Returns:
            A dictionary containing the feature's configuration if the key is found,
            otherwise `None`. Returns `None` if the registry was not loaded successfully.
        """
        if self._registry_data is None:
            logger.warning("Attempted to get feature definition, but registry is not loaded.")
            return None

        definition = self._registry_data.get(feature_key)
        if definition is None:
            logger.debug("Feature key '%s' not found in registry.", feature_key)
        return definition

    def get_all_feature_keys(self) -> List[str]:
        """
        Retrieves a list of all feature keys (top-level keys) defined in the registry.

        Returns:
            A list of strings representing all feature keys. Returns an empty list
            if the registry is not loaded or is empty.
        """
        if self._registry_data is None:
            logger.warning("Attempted to get all feature keys, but registry is not loaded.")
            return []
        return list(self._registry_data.keys())

    def get_output_properties(self, feature_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the 'output_properties' dictionary for a given feature key.

        Args:
            feature_key: The unique key of the feature.

        Returns:
            The 'output_properties' dictionary if the feature and its properties are found,
            otherwise `None`.
        """
        definition = self.get_feature_definition(feature_key)
        if definition and isinstance(definition.get("output_properties"), dict):
            return definition["output_properties"]
        if definition:
            logger.debug("Feature '%s' found, but 'output_properties' missing or not a dict.", feature_key)
        return None

    def get_calculator_type(self, feature_key: str) -> Optional[str]:
        """
        Retrieves the 'calculator_type' for a given feature key.

        Args:
            feature_key: The unique key of the feature.

        Returns:
            The 'calculator_type' string if the feature and its type are found,
            otherwise `None`.
        """
        definition = self.get_feature_definition(feature_key)
        if definition and isinstance(definition.get("calculator_type"), str):
            return definition["calculator_type"]
        if definition:
             logger.debug("Feature '%s' found, but 'calculator_type' missing or not a string.", feature_key)
        return None

    def get_parameters(self, feature_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the 'parameters' dictionary for a given feature key.

        Args:
            feature_key: The unique key of the feature.

        Returns:
            The 'parameters' dictionary if the feature and its parameters are found,
            otherwise `None`. Returns an empty dict if 'parameters' is explicitly null
            or not a dict in the definition.
        """
        definition = self.get_feature_definition(feature_key)
        if definition:
            params = definition.get("parameters")
            if isinstance(params, dict):
                return params
            if params is None: # If parameters is explicitly null or not set
                return {}
            logger.debug("Feature '%s' found, but 'parameters' is not a dictionary (found type: %s).", feature_key, type(params).__name__)
        return None

    def is_loaded(self) -> bool:
        """Checks if the registry data has been successfully loaded."""
        return self._registry_data is not None and len(self._registry_data) > 0

    def reload_registry(self) -> None:
        """Forces a reload of the feature registry data from the file."""
        logger.info("Reloading feature registry from %s", self.registry_path)
        self._load_registry()

if __name__ == '__main__':
    # Example Usage (assuming config/feature_registry.yaml exists and is populated)
    logging.basicConfig(level=logging.INFO)

    # Create a dummy config/feature_registry.yaml for this example to run
    example_registry_content = """
rsi_14_default:
  description: "Default 14-period Relative Strength Index."
  calculator_type: "rsi"
  input_type: "close_series"
  parameters: {"period": 14}
  output_properties: {"value_type": "float", "range": [0, 100]}
  version: "1.0"

macd_default:
  description: "Default MACD (12, 26, 9)."
  calculator_type: "macd"
  input_type: "close_series"
  parameters: {"fast": 12, "slow": 26, "signal": 9}
  version: "1.0"
"""
    example_registry_dir = Path("config")
    example_registry_dir.mkdir(exist_ok=True)
    with open(example_registry_dir / "feature_registry.yaml", "w") as f:
        f.write(example_registry_content)

    client = FeatureRegistryClient() # Uses default path

    if client.is_loaded():
        logger.info("Registry loaded successfully for example.")
        all_keys = client.get_all_feature_keys()
        logger.info(f"All feature keys: {all_keys}")

        rsi_def = client.get_feature_definition("rsi_14_default")
        if rsi_def:
            logger.info(f"RSI Definition: {rsi_def}")
            logger.info(f"RSI Output Properties: {client.get_output_properties('rsi_14_default')}")
            logger.info(f"RSI Calculator Type: {client.get_calculator_type('rsi_14_default')}")
            logger.info(f"RSI Parameters: {client.get_parameters('rsi_14_default')}")

        macd_params = client.get_parameters("macd_default")
        if macd_params is not None: # Note: get_parameters returns {} if params are null/missing
             logger.info(f"MACD Parameters: {macd_params}")

        non_existent_def = client.get_feature_definition("non_existent_feature")
        logger.info(f"Non-existent feature: {non_existent_def}")

        # Clean up dummy file
        # (Path(example_registry_dir / "feature_registry.yaml")).unlink()
        # (example_registry_dir).rmdir() # Careful if config dir is used for real things
    else:
        logger.error("Registry failed to load for example.")

    # Example of client with a non-existent file
    # non_existent_client = FeatureRegistryClient("config/does_not_exist.yaml")
    # logger.info(f"Is non_existent_client loaded? {non_existent_client.is_loaded()}")
    # logger.info(f"Keys from non_existent_client: {non_existent_client.get_all_feature_keys()}")

```
