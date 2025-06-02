"""Client for accessing and retrieving feature definitions from a YAML-based Feature Registry.

This module provides a `FeatureRegistryClient` class that handles loading, parsing,
and providing access to feature definitions stored in a YAML file. It is designed
to be a reusable component for any service that needs to query the feature registry.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

# Configure a logger for this module
logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = "config/feature_registry.yaml"

class RegistryLoadError(Exception):
    """Custom exception raised when the feature registry fails to load or parse."""

class FeatureRegistryClient:
    """A client for loading and accessing feature definitions from a YAML registry file.

    This client is responsible for parsing a YAML file that contains definitions
    for various features that can be computed by the `FeatureEngine`. It provides
    methods to query these definitions, such as retrieving a specific feature's
    configuration, listing all available feature keys, or getting specific
    attributes like parameters or output properties.

    The client attempts to load the registry upon instantiation. If the specified
    registry file is not found or contains invalid YAML, the client logs an error
    and operates in a degraded state where its methods will typically return `None`
    or empty collections, rather than raising an exception during query methods.
    The `is_loaded()` method can be used to check the status.

    Example:
        Assuming `config/feature_registry.yaml` exists with content like:
        ```yaml
        rsi_14_default:
          calculator_type: "rsi"
          parameters: {"period": 14}
          # ... other properties
        macd_default:
          calculator_type: "macd"
          # ... other properties
        ```

        ```python
        client = FeatureRegistryClient() # Uses default path
        if client.is_loaded():
            rsi_def = client.get_feature_definition("rsi_14_default")
            if rsi_def:
                print(f"RSI Calculator: {rsi_def.get('calculator_type')}")

            all_keys = client.get_all_feature_keys()
            print(f"All defined feature keys: {all_keys}")
        else:
            print("Feature registry could not be loaded.")
        ```

    Attributes:
        registry_path (Path): The path to the YAML feature registry file.
    """

    def __init__(self, registry_path: Path | str = DEFAULT_REGISTRY_PATH) -> None:
        """Initializes the FeatureRegistryClient and loads data from the registry.

        Args:
            registry_path: The path to the YAML feature registry file.
                           Defaults to `config/feature_registry.yaml`.
        """
        self.registry_path: Path = Path(registry_path)
        self._registry_data: dict[str, Any] | None = None
        self._load_registry()

    def _load_registry(self) -> None:
        """Loads feature definitions from the YAML file specified by `self.registry_path`.

        This method is called during `__init__`. It populates `self._registry_data`
        with the loaded content from the YAML file.

        Error Handling:
        - If the registry file is not found (`FileNotFoundError`), it logs an error
          and sets `self._registry_data` to an empty dictionary.
        - If the file content is not valid YAML (`yaml.YAMLError`), it logs an error
          and sets `self._registry_data` to an empty dictionary.
        - If the loaded YAML content is not a dictionary at its root, it logs an error
          and sets `self._registry_data` to an empty dictionary.
        - Other `Exception`s during loading are also caught, logged, and result in an
          empty registry data state.

        In these error scenarios, the client operates in a "degraded" mode, where
        query methods will typically return `None` or empty collections. This allows
        the application to continue running, albeit potentially without full feature
        capabilities if features depend on a valid registry.
        """
        try:
            if not self.registry_path.exists():
                logger.error("Feature registry file not found: %s", self.registry_path)
                self._registry_data = {} # Operate in a degraded state
                # Optionally raise RegistryLoadError("Feature registry file not found.")
                return

            with self.registry_path.open("r") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                logger.error(
                    "Feature registry content is not a dictionary. File: %s",
                    self.registry_path,
                )
                self._registry_data = {}
                # Optionally raise RegistryLoadError("Invalid registry format: not a dictionary.")
                return

            self._registry_data = data
            logger.info(
                "Feature registry loaded successfully from %s. Found %d definitions.",
                self.registry_path, len(self._registry_data),
            )

        except yaml.YAMLError as e:
            logger.exception(
                "Error parsing YAML in feature registry %s",
                self.registry_path,
                exc_info=e,
            )
            self._registry_data = {}
            # Optionally raise RegistryLoadError(f"YAML parsing error: {e}")
        except Exception as e: # Catch any other unexpected errors during file I/O or loading
            logger.exception(
                "Unexpected error loading feature registry %s",
                self.registry_path,
                exc_info=e,
            )
            self._registry_data = {}
            # Optionally raise RegistryLoadError(f"Unexpected loading error: {e}")

    def get_feature_definition(self, feature_key: str) -> dict[str, Any] | None:
        """Retrieves the raw configuration dictionary for a given feature key.

        Args:
            feature_key: The unique key of the feature as defined in the registry
                         (e.g., "rsi_14_default").

        Returns:
            A dictionary containing all configuration details for the specified
            `feature_key` if it exists in the loaded registry.
            Returns `None` if the `feature_key` is not found or if the registry
            data itself (`self._registry_data`) is not loaded (e.g., due to an
            error during `_load_registry`).
        """
        if self._registry_data is None:
            logger.warning(
                "Attempted to get feature definition ('%s'), but registry is not loaded.",
                feature_key,
            )
            return None

        definition = self._registry_data.get(feature_key)
        if definition is None:
            logger.debug("Feature key '%s' not found in registry.", feature_key)
        return definition

    def get_all_feature_keys(self) -> list[str]:
        """Retrieves a list of all top-level feature keys defined in the registry.

        Returns:
            A list of strings, where each string is a unique feature key from the
            registry. Returns an empty list if the registry is not loaded or if it
            contains no feature definitions.
        """
        if self._registry_data is None:
            logger.warning("Attempted to get all feature keys, but registry is not loaded.")
            return []
        return list(self._registry_data.keys())

    def get_output_properties(self, feature_key: str) -> dict[str, Any] | None:
        """Retrieves the 'output_properties' dictionary for a given feature key.

        This typically includes metadata about the feature's output, such as
        its value type, range, or if it's multidimensional.

        Args:
            feature_key: The unique key of the feature.

        Returns:
            The 'output_properties' dictionary if the feature exists and this
            key is defined as a dictionary within its definition.
            Returns `None` if the feature is not found, if 'output_properties'
            is missing, or if it's not a dictionary.
        """
        definition = self.get_feature_definition(feature_key)
        if definition and isinstance(definition.get("output_properties"), dict):
            return definition["output_properties"]  # type: ignore[no-any-return]
        if definition:
            logger.debug(
                "Feature '%s' found, but 'output_properties' missing or not a dict.",
                feature_key,
            )
        return None

    def get_calculator_type(self, feature_key: str) -> str | None:
        """Retrieves the 'calculator_type' string for a given feature key.

        This string usually identifies the specific calculation logic or class
        responsible for computing the feature (e.g., "rsi", "macd", "custom_indicator").

        Args:
            feature_key: The unique key of the feature.

        Returns:
            The 'calculator_type' as a string if the feature exists and this key
            is defined as a string within its definition.
            Returns `None` if the feature is not found, if 'calculator_type'
            is missing, or if it's not a string.
        """
        definition = self.get_feature_definition(feature_key)
        if definition and isinstance(definition.get("calculator_type"), str):
            return definition["calculator_type"]  # type: ignore[no-any-return]
        if definition:
            logger.debug(
                "Feature '%s' found, but 'calculator_type' missing or not a string.",
                feature_key,
            )
        return None

    def get_parameters(self, feature_key: str) -> dict[str, Any] | None:
        """Retrieves the 'parameters' dictionary for a given feature key.

        These parameters are typically passed to the feature's calculation logic.

        Args:
            feature_key: The unique key of the feature.

        Returns:
            A dictionary of parameters if the feature exists and 'parameters' is
            defined as a dictionary.
            Returns an empty dictionary if 'parameters' is explicitly set to `null`
            in the YAML or is not defined under the feature_key.
            Returns `None` if the feature_key itself is not found or if 'parameters'
            is present but not a dictionary (an error case logged by the method).
        """
        definition = self.get_feature_definition(feature_key)
        if definition:
            params = definition.get("parameters")
            if isinstance(params, dict):
                return params
            if params is None: # If parameters is explicitly null or not set
                return {}
            logger.debug(
                "Feature '%s' has 'parameters' defined, but it's not a dictionary "
                "(found type: %s).",
                feature_key, type(params).__name__,
            )
        return None

    def is_loaded(self) -> bool:
        """Checks if the registry data has been successfully loaded and contains content.

        Returns:
            True if `_registry_data` is a non-empty dictionary, False otherwise.
            A False return indicates that the registry file might be missing, empty,
            or improperly formatted, and the client is operating in a degraded state.
        """
        return self._registry_data is not None and len(self._registry_data) > 0

    def reload_registry(self) -> None:
        """Forces a reload of the feature registry data from the file.

        This can be used to pick up changes to the registry file specified
        during client initialization without restarting the application.
        """
        logger.info("Reloading feature registry from %s", self.registry_path)
        self._load_registry()
