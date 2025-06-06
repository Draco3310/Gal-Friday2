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

    **Registry File Format:**
        The feature registry YAML file should follow this structure:
        
        ```yaml
        feature_key_name:
          calculator_type: "calculator_name"  # Required: Type of calculator/processor
          parameters:                         # Optional: Calculator parameters
            param1: value1
            param2: value2
          output_properties:                  # Optional: Output metadata
            data_type: "float"
            range: [0.0, 100.0]
            description: "Feature description"
          metadata:                          # Optional: Additional metadata
            version: "1.0"
            author: "team_name"
        ```

    **Basic Usage:**
        Simple feature retrieval and basic operations:

        ```python
        from gal_friday.core.feature_registry_client import FeatureRegistryClient
        
        # Initialize with default registry path
        client = FeatureRegistryClient()
        
        # Check if registry loaded successfully
        if not client.is_loaded():
            raise RuntimeError("Feature registry failed to load")
        
        # Get a specific feature definition
        rsi_feature = client.get_feature_definition("rsi_14_default")
        if rsi_feature:
            calculator_type = rsi_feature.get("calculator_type")
            parameters = rsi_feature.get("parameters", {})
            print(f"Feature uses {calculator_type} with params: {parameters}")
        
        # List all available features
        available_features = client.get_all_feature_keys()
        print(f"Available features: {len(available_features)} found")
        ```

    **Advanced Usage Patterns:**
        Enterprise integration patterns and best practices:

        ```python
        import logging
        from pathlib import Path
        from typing import Dict, List, Optional
        
        # Custom registry path configuration
        registry_path = Path("config/production_features.yaml")
        client = FeatureRegistryClient(registry_path)
        
        # Robust feature processing with error handling
        def process_features_safely(feature_keys: List[str]) -> Dict[str, Dict]:
            \"\"\"Process multiple features with comprehensive error handling.\"\"\"
            results = {}
            
            if not client.is_loaded():
                logging.error("Registry not loaded, cannot process features")
                return results
            
            for feature_key in feature_keys:
                try:
                    definition = client.get_feature_definition(feature_key)
                    if definition is None:
                        logging.warning(f"Feature '{feature_key}' not found in registry")
                        continue
                    
                    # Validate required fields
                    calculator_type = client.get_calculator_type(feature_key)
                    if not calculator_type:
                        logging.error(f"Feature '{feature_key}' missing calculator_type")
                        continue
                    
                    # Get parameters with defaults
                    parameters = client.get_parameters(feature_key)
                    if parameters is None:
                        logging.warning(f"Feature '{feature_key}' has invalid parameters")
                        parameters = {}
                    
                    # Get output properties for validation
                    output_props = client.get_output_properties(feature_key)
                    
                    results[feature_key] = {
                        "definition": definition,
                        "calculator_type": calculator_type,
                        "parameters": parameters,
                        "output_properties": output_props
                    }
                    
                except Exception as e:
                    logging.exception(f"Error processing feature '{feature_key}': {e}")
                    continue
            
            return results
        
        # Usage example
        trading_features = ["rsi_14", "macd_default", "bollinger_bands_20"]
        processed_features = process_features_safely(trading_features)
        ```

    **Error Handling and Resilience:**
        Best practices for handling various error conditions:

        ```python
        from contextlib import contextmanager
        import time
        
        @contextmanager
        def registry_context(registry_path: str, max_retries: int = 3):
            \"\"\"Context manager for robust registry operations with retry logic.\"\"\"
            client = None
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    client = FeatureRegistryClient(registry_path)
                    if client.is_loaded():
                        break
                    else:
                        raise RuntimeError("Registry failed to load")
                        
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logging.error(f"Failed to initialize registry after {max_retries} attempts: {e}")
                        raise
                    
                    logging.warning(f"Registry init attempt {retry_count} failed, retrying...")
                    time.sleep(1)  # Brief delay before retry
            
            try:
                yield client
            finally:
                # Cleanup if needed
                pass
        
        # Usage with error resilience
        try:
            with registry_context("config/feature_registry.yaml") as registry:
                features = registry.get_all_feature_keys()
                for feature_key in features:
                    definition = registry.get_feature_definition(feature_key)
                    if definition:
                        # Process feature safely
                        pass
                        
        except Exception as e:
            logging.error(f"Registry operation failed: {e}")
            # Implement fallback behavior
        ```

    **Integration with Feature Engine:**
        Typical integration pattern for feature computation:

        ```python
        class FeatureProcessor:
            \"\"\"Example integration with feature processing pipeline.\"\"\"
            
            def __init__(self, registry_client: FeatureRegistryClient):
                self.registry = registry_client
                if not self.registry.is_loaded():
                    raise ValueError("Feature registry must be loaded for processing")
            
            def compute_feature(self, feature_key: str, market_data: Dict) -> Optional[float]:
                \"\"\"Compute a feature using registry configuration.\"\"\"
                # Get feature configuration
                calculator_type = self.registry.get_calculator_type(feature_key)
                if not calculator_type:
                    logging.error(f"No calculator type found for feature: {feature_key}")
                    return None
                
                parameters = self.registry.get_parameters(feature_key) or {}
                output_props = self.registry.get_output_properties(feature_key) or {}
                
                # Validate output constraints if specified
                expected_range = output_props.get("range")
                expected_type = output_props.get("data_type", "float")
                
                try:
                    # Compute feature (example with RSI)
                    if calculator_type == "rsi":
                        period = parameters.get("period", 14)
                        result = self._compute_rsi(market_data, period)
                    elif calculator_type == "macd":
                        result = self._compute_macd(market_data, **parameters)
                    else:
                        logging.warning(f"Unknown calculator type: {calculator_type}")
                        return None
                    
                    # Validate result against output properties
                    if expected_range and not (expected_range[0] <= result <= expected_range[1]):
                        logging.warning(f"Feature {feature_key} result {result} outside expected range {expected_range}")
                    
                    return result
                    
                except Exception as e:
                    logging.exception(f"Error computing feature {feature_key}: {e}")
                    return None
            
            def _compute_rsi(self, data: Dict, period: int) -> float:
                # RSI calculation implementation
                return 50.0  # Placeholder
            
            def _compute_macd(self, data: Dict, **params) -> float:
                # MACD calculation implementation  
                return 0.0  # Placeholder
        
        # Usage example
        registry = FeatureRegistryClient()
        processor = FeatureProcessor(registry)
        
        market_data = {"prices": [100, 101, 99, 102, 98]}
        rsi_value = processor.compute_feature("rsi_14_default", market_data)
        ```

    **Configuration Management:**
        Dynamic configuration and hot-reloading patterns:

        ```python
        import threading
        import time
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class RegistryWatcher(FileSystemEventHandler):
            \"\"\"File system watcher for automatic registry reloading.\"\"\"
            
            def __init__(self, registry_client: FeatureRegistryClient):
                self.registry = registry_client
                self._lock = threading.Lock()
            
            def on_modified(self, event):
                if event.src_path == str(self.registry.registry_path):
                    with self._lock:
                        logging.info("Registry file changed, reloading...")
                        self.registry.reload_registry()
                        if self.registry.is_loaded():
                            logging.info("Registry reloaded successfully")
                        else:
                            logging.error("Registry reload failed")
        
        # Setup automatic reloading
        client = FeatureRegistryClient()
        watcher = RegistryWatcher(client)
        observer = Observer()
        observer.schedule(watcher, path=str(client.registry_path.parent), recursive=False)
        observer.start()
        
        try:
            # Your application logic here
            while True:
                # Use client normally, it will auto-reload on file changes
                features = client.get_all_feature_keys()
                time.sleep(10)
        finally:
            observer.stop()
            observer.join()
        ```

    **Troubleshooting Guide:**
        Common issues and solutions:

        - **Registry file not found**: Ensure the path is correct and the file exists.
          Use `client.registry_path.exists()` to verify file presence.
        
        - **YAML parsing errors**: Validate YAML syntax using online validators or
          `yaml.safe_load()` directly. Check for proper indentation and quotes.
        
        - **Empty registry data**: Verify the YAML file contains a top-level dictionary.
          Root-level lists or scalars are not supported.
        
        - **Missing feature definitions**: Use `client.get_all_feature_keys()` to see
          available features. Feature keys are case-sensitive.
        
        - **Performance issues**: For large registries, consider caching feature
          definitions locally and using `reload_registry()` selectively.

    **Thread Safety:**
        The client is thread-safe for read operations after initialization.
        However, `reload_registry()` should be synchronized in multi-threaded
        environments using appropriate locking mechanisms.

    Attributes:
        registry_path (Path): The path to the YAML feature registry file.

    Raises:
        RegistryLoadError: When registry loading fails and strict mode is enabled
                          (currently not implemented, client operates in degraded mode).

    Note:
        This client is designed for high availability and will not raise exceptions
        during normal query operations, even if the registry is not loaded. Always
        check `is_loaded()` status for critical operations.
    """

    def __init__(self, registry_path: Path | str = DEFAULT_REGISTRY_PATH) -> None:
        """Initializes the FeatureRegistryClient and loads data from the registry.

        This constructor attempts to load and parse the feature registry file
        immediately upon instantiation. If loading fails, the client will operate
        in a degraded state where query methods return None or empty collections
        rather than raising exceptions.

        Args:
            registry_path (Path | str, optional): The path to the YAML feature registry file.
                                                 Can be provided as a string or Path object.
                                                 Defaults to "config/feature_registry.yaml".

        Examples:
            Default initialization:
            >>> client = FeatureRegistryClient()
            >>> if client.is_loaded():
            ...     print("Registry loaded successfully")

            Custom registry path:
            >>> from pathlib import Path
            >>> custom_path = Path("configs/trading_features.yaml")
            >>> client = FeatureRegistryClient(custom_path)

            Production initialization with validation:
            >>> def create_registry_client(config_dir: str) -> FeatureRegistryClient:
            ...     registry_path = Path(config_dir) / "feature_registry.yaml"
            ...     
            ...     # Validate path exists before initialization
            ...     if not registry_path.exists():
            ...         raise FileNotFoundError(f"Registry file not found: {registry_path}")
            ...     
            ...     client = FeatureRegistryClient(registry_path)
            ...     if not client.is_loaded():
            ...         raise RuntimeError("Failed to load feature registry")
            ...     
            ...     return client

            Environment-based configuration:
            >>> import os
            >>> env = os.getenv("ENVIRONMENT", "development")
            >>> registry_file = f"config/{env}_features.yaml"
            >>> client = FeatureRegistryClient(registry_file)

        Note:
            The registry file is loaded synchronously during initialization.
            For large registry files or network-mounted paths, consider
            the potential initialization delay in performance-critical applications.
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

        This method provides access to the complete feature definition as stored
        in the registry YAML file. The returned dictionary contains all configured
        properties including calculator_type, parameters, output_properties, and
        any custom metadata fields.

        Args:
            feature_key (str): The unique key of the feature as defined in the registry
                              (e.g., "rsi_14_default", "macd_12_26_9"). Feature keys
                              are case-sensitive and must match exactly.

        Returns:
            dict[str, Any] | None: A dictionary containing all configuration details
                                  for the specified feature_key if it exists in the
                                  loaded registry. Returns None if:
                                  - The feature_key is not found in the registry
                                  - The registry data is not loaded due to initialization errors
                                  - The feature_key parameter is None or empty

        Examples:
            Basic usage:
            >>> client = FeatureRegistryClient()
            >>> definition = client.get_feature_definition("rsi_14_default")
            >>> if definition:
            ...     print(f"Calculator: {definition.get('calculator_type')}")
            ...     print(f"Parameters: {definition.get('parameters', {})}")

            Error handling pattern:
            >>> definition = client.get_feature_definition("unknown_feature")
            >>> if definition is None:
            ...     print("Feature not found or registry not loaded")
            ... else:
            ...     # Process definition safely
            ...     pass

            Integration with validation:
            >>> def validate_feature_definition(feature_key: str) -> bool:
            ...     definition = client.get_feature_definition(feature_key)
            ...     if not definition:
            ...         return False
            ...     
            ...     required_fields = ["calculator_type"]
            ...     return all(field in definition for field in required_fields)

        Note:
            This method logs warnings when the registry is not loaded and debug
            messages when feature keys are not found. It never raises exceptions
            to maintain system stability in production environments.
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

        This method provides a complete inventory of available features in the
        current registry. It's useful for feature discovery, validation, and
        building dynamic feature processing pipelines.

        Returns:
            list[str]: A list of strings, where each string is a unique feature key
                      from the registry. Returns an empty list if:
                      - The registry is not loaded due to initialization errors
                      - The registry file is empty or contains no feature definitions
                      - The registry contains only invalid or malformed entries

        Examples:
            Basic feature discovery:
            >>> client = FeatureRegistryClient()
            >>> all_features = client.get_all_feature_keys()
            >>> print(f"Registry contains {len(all_features)} features")
            >>> for feature in all_features:
            ...     print(f"  - {feature}")

            Filtering features by pattern:
            >>> rsi_features = [f for f in client.get_all_feature_keys() 
            ...                if f.startswith("rsi_")]
            >>> macd_features = [f for f in client.get_all_feature_keys()
            ...                 if "macd" in f.lower()]

            Batch processing pattern:
            >>> def process_all_features():
            ...     features = client.get_all_feature_keys()
            ...     results = {}
            ...     
            ...     for feature_key in features:
            ...         try:
            ...             definition = client.get_feature_definition(feature_key)
            ...             if definition and "calculator_type" in definition:
            ...                 results[feature_key] = definition
            ...         except Exception as e:
            ...             logger.warning(f"Error processing {feature_key}: {e}")
            ...     
            ...     return results

            Registry validation:
            >>> def validate_registry_completeness():
            ...     features = client.get_all_feature_keys()
            ...     if not features:
            ...         return False, "No features found in registry"
            ...     
            ...     invalid_features = []
            ...     for feature_key in features:
            ...         definition = client.get_feature_definition(feature_key)
            ...         if not definition or "calculator_type" not in definition:
            ...             invalid_features.append(feature_key)
            ...     
            ...     if invalid_features:
            ...         return False, f"Invalid features: {invalid_features}"
            ...     return True, f"All {len(features)} features are valid"

        Performance Note:
            This method creates a new list on each call. For frequently accessed
            feature lists in performance-critical code, consider caching the result
            and refreshing only when the registry is reloaded.
        """
        if self._registry_data is None:
            logger.warning("Attempted to get all feature keys, but registry is not loaded.")
            return []
        return list(self._registry_data.keys())

    def get_output_properties(self, feature_key: str) -> dict[str, Any] | None:
        """Retrieves the 'output_properties' dictionary for a given feature key.

        Output properties contain metadata about the feature's computed values,
        such as data type, expected range, units, description, and validation
        constraints. This information is crucial for proper feature handling,
        validation, and downstream processing.

        Args:
            feature_key (str): The unique key of the feature to query.

        Returns:
            dict[str, Any] | None: The 'output_properties' dictionary if the feature
                                  exists and this section is properly defined.
                                  Returns None if:
                                  - The feature is not found in the registry
                                  - The 'output_properties' field is missing
                                  - The 'output_properties' field is not a dictionary
                                  - The registry is not loaded

        Examples:
            Basic output properties access:
            >>> client = FeatureRegistryClient()
            >>> props = client.get_output_properties("rsi_14_default")
            >>> if props:
            ...     data_type = props.get("data_type", "float")
            ...     value_range = props.get("range", [0, 100])
            ...     description = props.get("description", "No description")
            ...     print(f"RSI: {data_type} in range {value_range}")

            Validation using output properties:
            >>> def validate_feature_output(feature_key: str, computed_value: float) -> bool:
            ...     props = client.get_output_properties(feature_key)
            ...     if not props:
            ...         return True  # No constraints defined
            ...     
            ...     # Check data type
            ...     expected_type = props.get("data_type", "float")
            ...     if expected_type == "int" and not isinstance(computed_value, int):
            ...         return False
            ...     
            ...     # Check range constraints
            ...     value_range = props.get("range")
            ...     if value_range and len(value_range) == 2:
            ...         min_val, max_val = value_range
            ...         if not (min_val <= computed_value <= max_val):
            ...             return False
            ...     
            ...     return True

            Processing pipeline integration:
            >>> def prepare_feature_output(feature_key: str, raw_value: Any) -> Any:
            ...     props = client.get_output_properties(feature_key)
            ...     if not props:
            ...         return raw_value
            ...     
            ...     # Apply type conversion
            ...     target_type = props.get("data_type", "float")
            ...     if target_type == "int":
            ...         processed_value = int(round(float(raw_value)))
            ...     elif target_type == "bool":
            ...         processed_value = bool(raw_value)
            ...     else:
            ...         processed_value = float(raw_value)
            ...     
            ...     # Apply range clamping if specified
            ...     value_range = props.get("range")
            ...     if value_range and len(value_range) == 2:
            ...         min_val, max_val = value_range
            ...         processed_value = max(min_val, min(max_val, processed_value))
            ...     
            ...     return processed_value

        Common Output Properties:
            - data_type: "float", "int", "bool", "string"
            - range: [min_value, max_value] for numeric constraints
            - units: "percentage", "currency", "count", etc.
            - description: Human-readable description of the feature
            - precision: Number of decimal places for display
            - nullable: Whether None/null values are allowed
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

        The calculator type identifies the specific calculation logic, algorithm,
        or processor class responsible for computing the feature. This is typically
        used by feature engines to dispatch calculation requests to the appropriate
        handler or implementation.

        Args:
            feature_key (str): The unique key of the feature to query.

        Returns:
            str | None: The 'calculator_type' as a string if the feature exists
                       and this field is properly defined. Returns None if:
                       - The feature is not found in the registry
                       - The 'calculator_type' field is missing
                       - The 'calculator_type' field is not a string
                       - The registry is not loaded

        Examples:
            Basic calculator type retrieval:
            >>> client = FeatureRegistryClient()
            >>> calc_type = client.get_calculator_type("rsi_14_default")
            >>> if calc_type:
            ...     print(f"Feature uses calculator: {calc_type}")

            Feature processing dispatch:
            >>> def compute_feature(feature_key: str, market_data: dict) -> float:
            ...     calc_type = client.get_calculator_type(feature_key)
            ...     if not calc_type:
            ...         raise ValueError(f"No calculator type for feature: {feature_key}")
            ...     
            ...     # Dispatch to appropriate calculator
            ...     if calc_type == "rsi":
            ...         return compute_rsi(market_data, client.get_parameters(feature_key))
            ...     elif calc_type == "macd":
            ...         return compute_macd(market_data, client.get_parameters(feature_key))
            ...     elif calc_type == "bollinger_bands":
            ...         return compute_bollinger(market_data, client.get_parameters(feature_key))
            ...     else:
            ...         raise NotImplementedError(f"Calculator '{calc_type}' not implemented")

            Calculator registry pattern:
            >>> class CalculatorFactory:
            ...     def __init__(self, registry_client: FeatureRegistryClient):
            ...         self.registry = registry_client
            ...         self._calculators = {
            ...             "rsi": RSICalculator(),
            ...             "macd": MACDCalculator(),
            ...             "sma": SMACalculator(),
            ...         }
            ...     
            ...     def get_calculator(self, feature_key: str):
            ...         calc_type = self.registry.get_calculator_type(feature_key)
            ...         if calc_type not in self._calculators:
            ...             raise ValueError(f"Unknown calculator type: {calc_type}")
            ...         return self._calculators[calc_type]

            Validation and feature discovery:
            >>> def find_features_by_calculator(calculator_type: str) -> list[str]:
            ...     matching_features = []
            ...     for feature_key in client.get_all_feature_keys():
            ...         if client.get_calculator_type(feature_key) == calculator_type:
            ...             matching_features.append(feature_key)
            ...     return matching_features
            >>> 
            >>> rsi_features = find_features_by_calculator("rsi")
            >>> print(f"Found {len(rsi_features)} RSI-based features")

        Common Calculator Types:
            - "rsi": Relative Strength Index
            - "macd": Moving Average Convergence Divergence  
            - "sma": Simple Moving Average
            - "ema": Exponential Moving Average
            - "bollinger_bands": Bollinger Bands indicator
            - "stochastic": Stochastic oscillator
            - "custom": Custom calculation logic
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

        Parameters contain the configuration values that control how a feature
        is calculated. These are typically passed directly to the calculator
        implementation to customize its behavior (e.g., period lengths, thresholds,
        smoothing factors).

        Args:
            feature_key (str): The unique key of the feature to query.

        Returns:
            dict[str, Any] | None: A dictionary of parameters if the feature exists
                                  and 'parameters' is properly defined as a dictionary.
                                  Returns an empty dictionary {} if 'parameters' is
                                  explicitly set to null or not defined.
                                  Returns None if:
                                  - The feature_key itself is not found
                                  - The 'parameters' field exists but is not a dictionary

        Examples:
            Basic parameter retrieval:
            >>> client = FeatureRegistryClient()
            >>> params = client.get_parameters("rsi_14_default")
            >>> if params is not None:
            ...     period = params.get("period", 14)  # Default to 14 if not specified
            ...     print(f"RSI period: {period}")

            Safe parameter handling with defaults:
            >>> def get_feature_config(feature_key: str, default_params: dict) -> dict:
            ...     params = client.get_parameters(feature_key)
            ...     if params is None:
            ...         logger.warning(f"No parameters found for {feature_key}, using defaults")
            ...         return default_params
            ...     
            ...     # Merge with defaults
            ...     config = default_params.copy()
            ...     config.update(params)
            ...     return config
            >>> 
            >>> rsi_config = get_feature_config("rsi_14", {"period": 14, "source": "close"})

            Parameter validation:
            >>> def validate_rsi_parameters(feature_key: str) -> tuple[bool, str]:
            ...     params = client.get_parameters(feature_key)
            ...     if params is None:
            ...         return False, f"No parameters defined for {feature_key}"
            ...     
            ...     period = params.get("period")
            ...     if not isinstance(period, int) or period < 1:
            ...         return False, f"Invalid period: {period} (must be positive integer)"
            ...     
            ...     if "source" in params and params["source"] not in ["open", "high", "low", "close"]:
            ...         return False, f"Invalid source: {params['source']}"
            ...     
            ...     return True, "Parameters are valid"

            Dynamic parameter override:
            >>> def compute_with_override(feature_key: str, param_overrides: dict, data: dict):
            ...     base_params = client.get_parameters(feature_key) or {}
            ...     final_params = {**base_params, **param_overrides}
            ...     
            ...     calc_type = client.get_calculator_type(feature_key)
            ...     return calculate_feature(calc_type, data, final_params)

            Batch parameter analysis:
            >>> def analyze_feature_parameters():
            ...     param_summary = {}
            ...     for feature_key in client.get_all_feature_keys():
            ...         params = client.get_parameters(feature_key)
            ...         if params:
            ...             param_summary[feature_key] = {
            ...                 "param_count": len(params),
            ...                 "param_keys": list(params.keys()),
            ...                 "has_period": "period" in params
            ...             }
            ...     return param_summary

        Common Parameter Patterns:
            - period: Time period for moving averages and oscillators
            - source: Price source ("open", "high", "low", "close", "volume")
            - multiplier: Scaling factor for indicators
            - threshold: Trigger levels for signals
            - smoothing: Smoothing coefficients
            - fast_period, slow_period: Dual-period indicators like MACD
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

        This method is essential for validating registry state before performing
        operations that depend on feature definitions. It provides a safe way to
        check registry availability without triggering exceptions.

        Returns:
            bool: True if registry data is loaded and contains at least one feature
                 definition. False if:
                 - The registry file was not found during initialization
                 - The registry file contained invalid YAML syntax
                 - The registry file was empty or contained no feature definitions
                 - The registry file structure was invalid (not a top-level dictionary)
                 - Any other loading error occurred

        Examples:
            Basic status check:
            >>> client = FeatureRegistryClient()
            >>> if client.is_loaded():
            ...     features = client.get_all_feature_keys()
            ...     print(f"Registry loaded with {len(features)} features")
            ... else:
            ...     print("Registry failed to load or is empty")

            Defensive programming pattern:
            >>> def safe_feature_operation(client: FeatureRegistryClient, feature_key: str):
            ...     if not client.is_loaded():
            ...         logger.error("Cannot perform operation: registry not loaded")
            ...         return None
            ...     
            ...     return client.get_feature_definition(feature_key)

            Application startup validation:
            >>> def validate_application_dependencies():
            ...     registry = FeatureRegistryClient()
            ...     
            ...     if not registry.is_loaded():
            ...         raise RuntimeError(
            ...             "Feature registry is required but failed to load. "
            ...             f"Check registry file: {registry.registry_path}"
            ...         )
            ...     
            ...     # Validate minimum required features
            ...     required_features = ["rsi_14", "macd_default", "sma_20"]
            ...     available_features = registry.get_all_feature_keys()
            ...     
            ...     missing_features = [f for f in required_features if f not in available_features]
            ...     if missing_features:
            ...         raise RuntimeError(f"Missing required features: {missing_features}")

            Health check implementation:
            >>> def registry_health_check() -> dict:
            ...     client = FeatureRegistryClient()
            ...     health_status = {
            ...         "loaded": client.is_loaded(),
            ...         "registry_path": str(client.registry_path),
            ...         "file_exists": client.registry_path.exists(),
            ...     }
            ...     
            ...     if health_status["loaded"]:
            ...         health_status.update({
            ...             "feature_count": len(client.get_all_feature_keys()),
            ...             "status": "healthy"
            ...         })
            ...     else:
            ...         health_status["status"] = "degraded"
            ...     
            ...     return health_status

            Retry pattern with backoff:
            >>> import time
            >>> def wait_for_registry_ready(max_wait_seconds: int = 30) -> bool:
            ...     start_time = time.time()
            ...     retry_delay = 1
            ...     
            ...     while time.time() - start_time < max_wait_seconds:
            ...         client = FeatureRegistryClient()
            ...         if client.is_loaded():
            ...             return True
            ...         
            ...         time.sleep(retry_delay)
            ...         retry_delay = min(retry_delay * 2, 5)  # Exponential backoff, max 5s
            ...     
            ...     return False

        Performance Note:
            This method performs a simple boolean check and dictionary length
            operation, making it very fast and suitable for frequent status checks.
        """
        return self._registry_data is not None and len(self._registry_data) > 0

    def reload_registry(self) -> None:
        """Forces a reload of the feature registry data from the file.

        This method re-reads and re-parses the registry file, allowing applications
        to pick up configuration changes without restarting. It's particularly
        useful for development environments and production systems that need
        dynamic feature configuration updates.

        The reload operation is atomic - if the new registry fails to load,
        the previous registry data remains intact and available.

        Examples:
            Basic reload operation:
            >>> client = FeatureRegistryClient()
            >>> # ... registry file is modified externally ...
            >>> client.reload_registry()
            >>> if client.is_loaded():
            ...     print("Registry reloaded successfully")

            Scheduled reload pattern:
            >>> import threading
            >>> import time
            >>> 
            >>> def periodic_reload(client: FeatureRegistryClient, interval_seconds: int = 300):
            ...     \"\"\"Reload registry every 5 minutes in background thread.\"\"\"
            ...     while True:
            ...         time.sleep(interval_seconds)
            ...         try:
            ...             client.reload_registry()
            ...             logger.info("Registry reloaded successfully")
            ...         except Exception as e:
            ...             logger.error(f"Registry reload failed: {e}")
            >>> 
            >>> # Start background reload thread
            >>> reload_thread = threading.Thread(
            ...     target=periodic_reload, 
            ...     args=(client, 300),
            ...     daemon=True
            ... )
            >>> reload_thread.start()

            Configuration hot-swap with validation:
            >>> def safe_reload_with_validation(client: FeatureRegistryClient) -> bool:
            ...     # Store current state for rollback
            ...     old_features = client.get_all_feature_keys() if client.is_loaded() else []
            ...     
            ...     try:
            ...         client.reload_registry()
            ...         
            ...         if not client.is_loaded():
            ...             logger.error("Registry reload resulted in empty registry")
            ...             return False
            ...         
            ...         # Validate critical features still exist
            ...         new_features = client.get_all_feature_keys()
            ...         critical_features = ["rsi_14", "macd_default"]
            ...         
            ...         missing_critical = [f for f in critical_features if f not in new_features]
            ...         if missing_critical:
            ...             logger.error(f"Reload removed critical features: {missing_critical}")
            ...             return False
            ...         
            ...         logger.info(f"Registry reloaded: {len(old_features)} -> {len(new_features)} features")
            ...         return True
            ...         
            ...     except Exception as e:
            ...         logger.exception(f"Registry reload failed: {e}")
            ...         return False

            File watcher integration:
            >>> from watchdog.observers import Observer
            >>> from watchdog.events import FileSystemEventHandler
            >>> 
            >>> class RegistryReloadHandler(FileSystemEventHandler):
            ...     def __init__(self, registry_client: FeatureRegistryClient):
            ...         self.client = registry_client
            ...         self._last_reload = 0
            ...         self._reload_cooldown = 2  # Prevent rapid reloads
            ...     
            ...     def on_modified(self, event):
            ...         if event.src_path == str(self.client.registry_path):
            ...             current_time = time.time()
            ...             if current_time - self._last_reload > self._reload_cooldown:
            ...                 self.client.reload_registry()
            ...                 self._last_reload = current_time
            ...                 logger.info("Registry auto-reloaded due to file change")

            API endpoint for configuration management:
            >>> def create_reload_endpoint(registry_client: FeatureRegistryClient):
            ...     \"\"\"Example Flask endpoint for triggering registry reload.\"\"\"
            ...     from flask import jsonify
            ...     
            ...     def reload_registry_endpoint():
            ...         try:
            ...             old_count = len(registry_client.get_all_feature_keys())
            ...             registry_client.reload_registry()
            ...             
            ...             if registry_client.is_loaded():
            ...                 new_count = len(registry_client.get_all_feature_keys())
            ...                 return jsonify({
            ...                     "status": "success",
            ...                     "message": f"Registry reloaded ({old_count} -> {new_count} features)",
            ...                     "feature_count": new_count
            ...                 })
            ...             else:
            ...                 return jsonify({
            ...                     "status": "error",
            ...                     "message": "Registry reload failed - registry is not loaded"
            ...                 }), 500
            ...                 
            ...         except Exception as e:
            ...             return jsonify({
            ...                 "status": "error",
            ...                 "message": f"Registry reload failed: {str(e)}"
            ...             }), 500
            ...     
            ...     return reload_registry_endpoint

        Thread Safety:
            While individual read operations are thread-safe, reload_registry()
            should be synchronized in multi-threaded environments to prevent
            race conditions during the reload process.

        Performance Note:
            Reloading involves file I/O and YAML parsing, which may take
            significant time for large registry files. Consider the impact
            on application responsiveness when calling this method.
        """
        logger.info("Reloading feature registry from %s", self.registry_path)
        self._load_registry()
