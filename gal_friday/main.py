#!/usr/bin/env python
"""
Main entry point for the Gal-Friday trading bot application.

This script initializes all necessary components (configuration, logging, services,
 event bus, executor), wires them together, starts the application, and handles
 graceful shutdown.
"""

import argparse  # Added for command-line argument parsing
import asyncio
import concurrent.futures
import functools
import logging
import logging.handlers  # Added for RotatingFileHandler
from pathlib import Path  # PTHxxx fix: Import Path
import signal
import sys
from typing import TYPE_CHECKING, Any, Optional, Union

# --- Custom Exceptions Import ---
from .exceptions import (
    ConfigurationLoadingFailedExit,
    DependencyMissingError,
    ExecutionHandlerInstantiationFailedExit,
    LoggerServiceInstantiationFailedExit,
    MarketPriceServiceCriticalFailureExit,
    MarketPriceServiceUnsupportedModeError,
    PortfolioManagerInstantiationFailedExit,
    PubSubManagerInstantiationFailedExit,
    PubSubManagerStartFailedExit,
    RiskManagerInstantiationFailedExit,
    UnsupportedModeError,
)

# Version information
__version__ = "0.1.0"  # Add version tracking

# Error messages
_CONFIG_NOT_INITIALIZED_MSG = "ConfigManager is not initialized"

# --- Conditional Imports for Type Checking --- #
if TYPE_CHECKING:
    # ERA001: Removed commented-out import.
    # Define a proper protocol/interface for execution handlers
    from typing import Protocol

    from .cli_service import CLIService as CLIServiceType
    from .config_manager import ConfigManager as ConfigManagerType
    from .core.pubsub import PubSubManager as PubSubManagerType
    from .data_ingestor import DataIngestor as DataIngestorType

    # Ensure these are imported if they are to be used as string literals in hints
    from .execution.kraken import KrakenExecutionHandler
    from .feature_engine import FeatureEngine as FeatureEngineType
    from .historical_data_service import HistoricalDataService as HistoricalDataServiceType
    from .logger_service import LoggerService as LoggerServiceType
    from .market_price_service import MarketPriceService as MarketPriceServiceType
    from .monitoring_service import MonitoringService as MonitoringServiceType
    from .portfolio_manager import PortfolioManager as PortfolioManagerType
    from .prediction_service import PredictionService as PredictionServiceType
    from .risk_manager import RiskManager as RiskManagerType
    from .simulated_execution_handler import SimulatedExecutionHandler
    from .strategy_arbitrator import StrategyArbitrator as StrategyArbitratorType

    class ExecutionHandlerProtocol(Protocol):
        """Protocol defining interface for execution handlers."""

        def __init__(
            self,
            *,
            config_manager: "ConfigManagerType",
            pubsub_manager: "PubSubManagerType",
            logger_service: "LoggerServiceType",
            **kwargs: object,
        ) -> None:
            """Initialize an execution handler.

            Args
            ----
                config_manager: Configuration manager instance
                pubsub_manager: Publish-subscribe manager instance
                logger_service: Logger service instance
                **kwargs: Additional keyword arguments
            """
            ...

        async def start(self) -> None:
            """Start the execution handler and initialize any connections."""
            ...

        async def stop(self) -> None:
            """Stop the execution handler and clean up resources."""
            ...

        # Add other common methods that execution handlers should implement
        def submit_order(self, order_data: dict[str, Any]) -> str:
            """Submit an order to the exchange.

            Args
            ----
                order_data: Dictionary containing order details

            Returns
            -------
                Order ID from the exchange
            """
            ...

        def cancel_order(self, order_id: str) -> bool:
            """Cancel an existing order.

            Args
            ----
                order_id: ID of the order to cancel

            Returns
            -------
                True if cancellation was successful, False otherwise
            """
            ...

    # Now use this protocol for type annotations
    ExecutionHandlerTypeHint = type[ExecutionHandlerProtocol]
    _ExecutionHandlerType = Union[KrakenExecutionHandler, SimulatedExecutionHandler]


# --- Attempt to import core application modules (Runtime) --- #
try:
    from .config_manager import ConfigManager
except ImportError:
    print("Failed to import ConfigManager")
    ConfigManager = None  # type: ignore[assignment,misc]

try:
    from .core.pubsub import PubSubManager
except ImportError:
    print("Failed to import PubSubManager")
    PubSubManager = None  # type: ignore[assignment,misc]

try:
    from .data_ingestor import DataIngestor
except ImportError:
    print("Failed to import DataIngestor")
    DataIngestor = None  # type: ignore[assignment,misc]

try:
    from .prediction_service import PredictionService
except ImportError:
    print("Failed to import PredictionService")
    PredictionService = None  # type: ignore[assignment,misc]

try:
    from .strategy_arbitrator import StrategyArbitrator
except ImportError:
    print("Failed to import StrategyArbitrator")
    StrategyArbitrator = None  # type: ignore[assignment,misc]

try:
    from .portfolio_manager import PortfolioManager
except ImportError:
    print("Failed to import PortfolioManager")
    PortfolioManager = None  # type: ignore[assignment,misc]

try:
    from .risk_manager import RiskManager
except ImportError:
    print("Failed to import RiskManager")
    RiskManager = None  # type: ignore[assignment,misc]

# --- Execution Handler Imports (Runtime) --- #
try:
    from .execution.kraken import KrakenExecutionHandler
except ImportError as e:
    print(f"Failed to import KrakenExecutionHandler: {e}")
    KrakenExecutionHandler = None  # type: ignore

try:
    from .simulated_execution_handler import SimulatedExecutionHandler
except ImportError:
    print("Failed to import SimulatedExecutionHandler")
    SimulatedExecutionHandler = None  # type: ignore

# --- Other Service Imports (Runtime) --- #
try:
    from .logger_service import LoggerService
except ImportError:
    print("Failed to import LoggerService")
    LoggerService = None  # type: ignore[assignment,misc]

try:
    from .monitoring_service import MonitoringService
except ImportError:
    print("Failed to import MonitoringService")
    MonitoringService = None  # type: ignore[assignment,misc]

try:
    from .cli_service import CLIService
except ImportError:
    print("Failed to import CLIService")
    CLIService = None  # type: ignore[assignment,misc]

try:
    from .market_price_service import MarketPriceService
except ImportError:
    print("Failed to import MarketPriceService")
    MarketPriceService = None  # type: ignore[assignment,misc]

try:
    from .historical_data_service import HistoricalDataService
except ImportError:
    print("Failed to import HistoricalDataService")
    HistoricalDataService = None  # type: ignore[assignment,misc]

# --- Import concrete service implementations --- #
try:
    from .market_price.kraken_service import KrakenMarketPriceService
except ImportError as e:
    print(f"Failed to import KrakenMarketPriceService: {e}")
    KrakenMarketPriceService = None  # type: ignore

try:
    from .kraken_historical_data_service import KrakenHistoricalDataService
except ImportError as e:
    print(f"Failed to import KrakenHistoricalDataService: {e}")
    KrakenHistoricalDataService = None  # type: ignore

try:
    from .simulated_market_price_service import (  # Restored runtime import
        SimulatedMarketPriceService,
    )
except ImportError:
    print("Failed to import SimulatedMarketPriceService")
    SimulatedMarketPriceService = None  # type: ignore # Restored fallback

# --- Global Setup --- #
# Basic logging configured immediately to catch early issues
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Log to console initially
)
log = logging.getLogger(__name__)

# Global shutdown event to signal termination across tasks
shutdown_event = asyncio.Event()
class ServiceInitializationError(RuntimeError):
    """Raised when a required service fails to initialize."""



class MarketPriceServiceError(ServiceInitializationError):
    """Raised when the market price service is not available."""



class PubSubManagerError(ServiceInitializationError):
    """Raised when the PubSub manager is not available."""



class LoggerServiceError(ServiceInitializationError):
    """Raised when the logger service is not available."""



class GlobalState:
    """Global application state to avoid using global variables."""

    main_event_loop: asyncio.AbstractEventLoop | None = None

# Global state instance
global_state = GlobalState()

# --- Logging Setup Function --- #
# Use string literal for the type hint
def setup_logging(
    config: Optional["ConfigManagerType"], log_level_override: str | None = None
) -> None:
    """Configure logging based on the application configuration."""
    # Runtime check still needed
    if config is None or ConfigManager is None:
        log.warning("ConfigManager instance or class not available, cannot configure logging.")
        return

    # No assertion needed here as we checked config is not None
    log_config = config.get("logging", {})
    log_level_name = (
        log_level_override or log_config.get("level", "INFO").upper()
    )  # Use override if provided
    log_level = getattr(logging, log_level_name, logging.INFO)

    root_logger = logging.getLogger()  # Get the root logger
    root_logger.setLevel(log_level)

    # Clear existing handlers (e.g., from basicConfig)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    log.info("Root logger level set to %s", log_level_name)

    # --- Console Handler --- #
    console_config = log_config.get("console", {})
    if console_config.get("enabled", True):
        console_format = console_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler = logging.StreamHandler(sys.stdout)
        # Handler level defaults to root logger level
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            console_format, datefmt=log_config.get("date_format")
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        log.info("Console logging enabled.")

    # --- JSON File Handler --- #
    json_file_config = log_config.get("json_file", {})
    if json_file_config.get("enabled", False):
        log_filename = json_file_config.get("filename")
        if log_filename:
            # Ensure log directory exists
            log_dir_path = Path(log_filename).parent
            if log_dir_path and not log_dir_path.exists():
                try:
                    log_dir_path.mkdir(parents=True, exist_ok=True)
                    log.info("Created log directory: %s", log_dir_path)
                except OSError:
                    log.exception("Could not create log directory %s", log_dir_path)
                    log_filename = None  # Prevent handler creation if dir fails

            if log_filename:
                max_bytes = json_file_config.get("max_bytes", 10 * 1024 * 1024)  # Default 10MB
                backup_count = json_file_config.get("backup_count", 5)
                # Note: Using standard formatter for now. For true JSON, need jsonlogger library.
                # Consider adding jsonlogger to requirements.txt and
                # implementing later.
                file_format = json_file_config.get(
                    "format", "%(asctime)s %(name)s %(levelname)s %(message)s"
                )

                file_handler = logging.handlers.RotatingFileHandler(
                    log_filename, maxBytes=max_bytes, backupCount=backup_count
                )
                file_handler.setLevel(log_level)
                file_formatter = logging.Formatter(
                    file_format, datefmt=log_config.get("date_format")
                )
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
                log.info("File logging enabled: %s", log_filename)
        else:
            log.warning("File logging enabled in config but no filename specified.")

    # --- Database Handler --- #
    db_config = log_config.get("database", {})
    if db_config.get("enabled", False):
        log.info("Database logging configured as enabled. LoggerService will handle setup.")
        # Actual DB handler setup is deferred to LoggerService.start()


# --- Graceful Shutdown Handler --- #
def handle_shutdown(sig: signal.Signals) -> None:
    """Set the shutdown event when a signal is received."""
    log.warning("Received shutdown signal: %s. Initiating graceful shutdown...", sig.name)
    shutdown_event.set()


# --- Main Application Class --- #
class GalFridayApp:
    """Encapsulates the main application logic and lifecycle."""

    def __init__(self) -> None:  # Add return type
        """Initialize application state attributes."""
        log.info("Initializing GalFridayApp...")
        # Use Optional['ClassName'] string literals for type hints
        self.config: ConfigManagerType | None = None
        self.pubsub: PubSubManagerType | None = None
        self.executor: concurrent.futures.ProcessPoolExecutor | None = None
        self.services: list[Any] = []  # UP006: List -> list; Use Any for now, can refine later
        self.running_tasks: list[asyncio.Task] = []  # UP006: List -> list
        self.args: argparse.Namespace | None = None

        # Store references to specific services after instantiation for DI
        self.logger_service: LoggerServiceType | None = None
        # Added
        self.market_price_service: MarketPriceServiceType | None = None
        # Added
        self.historical_data_service: HistoricalDataServiceType | None = None
        self.portfolio_manager: PortfolioManagerType | None = None
        # Use the type alias defined in TYPE_CHECKING
        self.execution_handler: _ExecutionHandlerType | None = None
        self.monitoring_service: MonitoringServiceType | None = None
        self.cli_service: CLIServiceType | None = None
        self.risk_manager: RiskManagerType | None = None
        self.data_ingestor: DataIngestorType | None = None
        self.feature_engine: FeatureEngineType | None = None
        self.prediction_service: PredictionServiceType | None = None
        self.strategy_arbitrator: StrategyArbitratorType | None = None

        # Keep a direct reference to config_manager if needed by other methods
        self._config_manager_instance: ConfigManagerType | None = None

    def _load_configuration(self, config_path: str) -> None:  # Accept config_path parameter
        """Load the application configuration."""
        try:
            self._ensure_class_available(ConfigManager, "ConfigManager", "Configuration loading")
            # Pass pubsub and loop to ConfigManager for dynamic reloading
            # Pubsub might not be initialized yet. Loop should be available from main_async.
            self._config_manager_instance = ConfigManager(
                config_path=config_path,
                logger_service=logging.getLogger(
                    "gal_friday.config_manager"
                ),  # Give it its own logger instance
                pubsub_manager=self.pubsub,  # self.pubsub will be set in _instantiate_pubsub
                loop=global_state.main_event_loop,
            )
            self.config = self._config_manager_instance  # Main access point for config
            log.info("Configuration loaded successfully from: %s", config_path)
            if not self.config.is_valid():
                log.error(
                    "Initial configuration is invalid. Review errors logged by ConfigManager."
                )
                # Depending on desired strictness, could raise ConfigurationLoadingFailedExit here.
        except Exception as e:
            log.exception(
                "FATAL: Failed to load configuration from %s",
                config_path
            )
            raise ConfigurationLoadingFailedExit from e

    def _setup_executor(self) -> None:  # Add return type
        """Set up the ProcessPoolExecutor."""
        if self.config is None or ConfigManager is None:  # Runtime checks
            log.error("Cannot setup executor without configuration.")
            self.executor = None
            return
        try:
            # No assertion needed due to check above
            max_workers = self.config.get_int("prediction_service.executor_workers", 1)
            if max_workers < 1:
                log.warning("Invalid executor_workers count (%s), defaulting to 1.", max_workers)
                max_workers = 1
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
            log.info("ProcessPoolExecutor created with max_workers=%s.", max_workers)
        except Exception:
            log.exception("ERROR: Failed to create ProcessPoolExecutor")
            self.executor = None

    def _handle_missing_config_for_pubsub(self) -> None:
        """Handle missing configuration for PubSubManager initialization."""
        error_msg = (
            "ConfigManager instance not available for PubSubManager. "
            "Load configuration first."
        )
        log.critical(error_msg)
        raise DependencyMissingError("PubSubManager", error_msg)

    def _update_config_manager_with_pubsub(self) -> None:
        """Update ConfigManager with PubSubManager instance if available."""
        if self._config_manager_instance and hasattr(
            self._config_manager_instance, "set_pubsub_manager"
        ):
            self._config_manager_instance.set_pubsub_manager(self.pubsub)
        if hasattr(self, "_logger"):
            self._logger.info("Updated ConfigManager with PubSubManager instance.")

    def _instantiate_pubsub(self) -> None:
        """Instantiate the PubSubManager."""
        try:
            self._ensure_class_available(
                PubSubManager, "PubSubManager", "PubSubManager instantiation"
            )

            # Should use self._config_manager_instance for initial state
            if self.config is None:
                # This state indicates _load_configuration wasn't called or
                # failed before pubsub init
                self._handle_missing_config_for_pubsub()
                # _handle_missing_config_for_pubsub raises an exception,
                # but type checker doesn't know that
                return

            if not hasattr(self.config, "get"):
                self._raise_missing_config_method("get")

            self.pubsub = PubSubManager(
                logger=logging.getLogger("gal_friday.pubsub"),
                config_manager=self.config,
            )
            log.info("PubSubManager instantiated successfully.")

            # Now that pubsub is available, if config_manager was already
            # created, update it through the proper setter if available
            self._update_config_manager_with_pubsub()
        except Exception as e:
            log.exception("FATAL: Failed to instantiate PubSubManager")
            raise PubSubManagerInstantiationFailedExit from e

    def _init_strategy_arbitrator(self) -> None:
        """Instantiate the StrategyArbitrator."""
        self._ensure_strategy_arbitrator_prerequisites()
        if self.config is None:
            self._raise_config_not_initialized()
        assert isinstance(self.config, ConfigManagerType)  # Add type guard
        strategy_arbitrator_config = self.config.get("strategy_arbitrator", {})
        if self.market_price_service is None:
            raise MarketPriceServiceError
        if self.pubsub is None:
            raise PubSubManagerError
        if self.logger_service is None:
            raise LoggerServiceError

        self.strategy_arbitrator = StrategyArbitrator(
            config=strategy_arbitrator_config,
            pubsub_manager=self.pubsub,
            logger_service=self.logger_service,
            market_price_service=self.market_price_service,
        )
        log.debug("StrategyArbitrator instantiated.")

    def _init_cli_service(self) -> None:
        """Instantiate the CLIService."""
        if CLIService is not None:
            # Prerequisite checks for CLIService (monitoring_service, portfolio_manager)
            if self.monitoring_service is None:
                raise DependencyMissingError(
                    component="CLIService", dependency="MonitoringService instance"
                )
            if self.portfolio_manager is None:
                raise DependencyMissingError(
                    component="CLIService", dependency="PortfolioManager instance"
                )

            if self.logger_service is None:
                raise LoggerServiceError

            self.cli_service = CLIService(
                monitoring_service=self.monitoring_service,
                logger_service=self.logger_service,
                main_app_controller=self,  # Pass self for app control
                portfolio_manager=self.portfolio_manager,
            )
            self.services.append(self.cli_service)
            log.debug("CLIService instantiated.")
        else:
            self.cli_service = None
            log.info("CLIService not available or not configured.")

    def _ensure_class_available(
        self,
        class_obj: type | None,
        class_name_str: str,
        required_by_component: str = "GalFridayApp",
    ) -> None:
        """Ensure a class is available (not None), raises DependencyMissingError if not."""
        if class_obj is None:
            raise DependencyMissingError(
                component=required_by_component,
                dependency=f"{class_name_str} class not available or import failed",
            )

    def _ensure_strategy_arbitrator_prerequisites(self) -> None:
        """Ensure all prerequisites for StrategyArbitrator are met before instantiation."""
        if self.market_price_service is None:
            raise DependencyMissingError(
                component="StrategyArbitrator", dependency="MarketPriceService instance"
            )
        if StrategyArbitrator is None:  # This is the class itself from the import block
            raise DependencyMissingError(
                component="StrategyArbitrator",
                dependency="StrategyArbitrator class (import failed or not available)",
            )

    def _ensure_risk_manager_prerequisites(self) -> None:
        """Ensure all prerequisites for RiskManager are met before instantiation."""
        if self.market_price_service is None:
            raise DependencyMissingError(
                component="RiskManager", dependency="MarketPriceService instance"
            )
        if self.portfolio_manager is None:
            raise DependencyMissingError(
                component="RiskManager", dependency="PortfolioManager instance"
            )
        if RiskManager is None:  # This is the class itself from the import block
            raise DependencyMissingError(
                component="RiskManager",
                dependency="RiskManager class (import failed or not available)",
            )

    def _ensure_portfolio_manager_prerequisites(self) -> None:
        """Ensure all prerequisites for PortfolioManager are met before instantiation."""
        if self.market_price_service is None:
            raise DependencyMissingError(
                component="PortfolioManager", dependency="MarketPriceService instance"
            )
        if PortfolioManager is None:  # This is the class itself
            raise DependencyMissingError(
                component="PortfolioManager",
                dependency="PortfolioManager class (import failed or not available)",
            )

    def _handle_risk_manager_none_after_init(self) -> None:
        """Handle the case where RiskManager is None after a successful init call."""
        # This method exists to abstract the raise for TRY301.
        raise RiskManagerInstantiationFailedExit(component_name="RiskManager")

    def _raise_missing_config_method(self, method_name: str) -> None:
        """Raise error for missing config method."""
        msg = f"ConfigManager does not have required '{method_name}' method"
        raise RuntimeError(msg)

    def _raise_config_not_initialized(self) -> None:
        """Raise error when config is not initialized."""
        raise RuntimeError(_CONFIG_NOT_INITIALIZED_MSG)

    def _raise_config_not_loaded_for_pubsub(self) -> None:
        """Raise DependencyMissingError if config is not loaded for PubSub init."""
        raise DependencyMissingError(
            component="PubSubManager instantiation",
            dependency="Configuration (self.config) not loaded",
        )

    def _raise_dependency_not_instantiated(
        self, component_name: str, dependency_name: str
    ) -> None:
        """Raise DependencyMissingError for a non-instantiated dependency."""
        raise DependencyMissingError(
            component=component_name, dependency=f"{dependency_name} not instantiated or available"
        )

    def _raise_logger_service_instantiation_failed(self) -> None:
        """Raise LoggerServiceInstantiationFailedExit."""
        raise LoggerServiceInstantiationFailedExit

    def _raise_kraken_market_price_service_unavailable_for_live_mode(self) -> None:
        """Raise DependencyMissingError for unavailable KrakenMarketPriceService in live mode."""
        raise DependencyMissingError(
            component="MarketPriceService (live mode)", dependency="KrakenMarketPriceService class"
        )

    def _raise_market_price_service_unsupported_mode(self, mode: str) -> None:
        """Raise MarketPriceServiceUnsupportedModeError."""
        # Note: supported_modes can be dynamically fetched or hardcoded if static
        raise MarketPriceServiceUnsupportedModeError(
            mode=mode,
            supported_modes=["live", "paper"],  # Example modes
        )

    def _raise_market_price_service_critical_failure(self) -> None:
        """Raise MarketPriceServiceCriticalFailureExit."""
        raise MarketPriceServiceCriticalFailureExit

    def _raise_portfolio_manager_instantiation_failed(self) -> None:
        """Raise PortfolioManagerInstantiationFailedExit."""
        raise PortfolioManagerInstantiationFailedExit

    def _instantiate_execution_handler(self, run_mode: str) -> _ExecutionHandlerType:
        """Instantiate the correct ExecutionHandler based on the run mode.

        Returns
        -------
            The instantiated execution handler.

        Raises
        ------
            ExecutionHandlerInstantiationFailedExit: If instantiation fails after attempting.
            DependencyMissingError: If a required component for the handler is missing.
            UnsupportedModeError: If the run_mode is not supported.
        """
        # Reset self.execution_handler to None at the start of instantiation attempt for
        # a given mode.
        # This ensures that if a previous mode set it, it's cleared before the new mode attempts.
        # However, typically this method is called once per app run with a determined mode.
        self.execution_handler = None

        # Runtime checks for required classes and instances
        # (these should raise if there's an issue)
        if self.config is None or ConfigManager is None:
            raise DependencyMissingError(component="ExecutionHandler", dependency="Config")
        if self.pubsub is None or PubSubManager is None:
            raise DependencyMissingError(component="ExecutionHandler", dependency="PubSub")
        if self.logger_service is None or LoggerService is None:
            raise DependencyMissingError(component="ExecutionHandler", dependency="LoggerService")
        if self.monitoring_service is None or MonitoringService is None:  # Added check
            raise DependencyMissingError(
                component="ExecutionHandler", dependency="MonitoringService"
            )

        if run_mode == "live":
            if KrakenExecutionHandler is None:
                raise DependencyMissingError(
                    component="Live mode ExecutionHandler",
                    dependency="KrakenExecutionHandler class",
                )
            self.execution_handler = KrakenExecutionHandler(
                config_manager=self.config,
                pubsub_manager=self.pubsub,
                logger_service=self.logger_service,
                monitoring_service=self.monitoring_service,
            )
            log.debug("KrakenExecutionHandler instantiated.")

        elif run_mode == "paper":
            if SimulatedExecutionHandler is None:
                raise DependencyMissingError(
                    component="Paper mode ExecutionHandler",
                    dependency="SimulatedExecutionHandler class",
                )
            if self.historical_data_service is None or HistoricalDataService is None:
                raise DependencyMissingError(
                    component="SimulatedExecutionHandler", dependency="HistoricalDataService"
                )
            self.execution_handler = SimulatedExecutionHandler(
                config_manager=self.config,
                pubsub_manager=self.pubsub,
                data_service=self.historical_data_service,
                logger_service=self.logger_service,
            )
            log.debug("SimulatedExecutionHandler instantiated for paper mode.")

        else:
            # This will raise UnsupportedModeError if run_mode is not 'live' or 'paper'
            raise UnsupportedModeError(mode=run_mode, supported_modes=["live", "paper"])

        # Final check: If after all mode logic, handler is still None, something is wrong.
        if self.execution_handler is None:
            # This path indicates a logic flaw if no specific error (DependencyMissing,
            # UnsupportedMode) was raised earlier.
            log.critical(
                "Execution handler is unexpectedly None after instantiation attempt for mode: %s.",
                run_mode,
            )
            raise ExecutionHandlerInstantiationFailedExit(mode=run_mode)

        # Append the successfully instantiated handler to the services list
        self.services.append(self.execution_handler)
        # We know execution_handler is not None at this point due to earlier checks
        assert self.execution_handler is not None, (
            "Execution handler should not be None at this point"
        )
        return self.execution_handler  # type: ignore[return-value]

    async def initialize(self, args: argparse.Namespace) -> None:  # Add return type
        """Load configuration, set up logging, and instantiate components."""
        log.info("Initializing GalFridayApp (Version: %s)...", __version__)
        self.args = args  # args should be guaranteed by main_async

        # --- 1. Configuration Loading ---
        self._load_configuration(args.config)
        # No assertion needed, _load_configuration raises SystemExit on failure

        # --- 2. Logging Setup ---
        try:
            # Pass the instance and log_level override
            setup_logging(self.config, args.log_level)
            log.info("Logging configured successfully.")
        except Exception:
            log.exception("ERROR: Failed to configure logging")
            # Continue running with basic logging if setup fails

        # --- 3. Executor Setup ---
        self._setup_executor()

        # --- 4. PubSub Manager Instantiation ---
        self._instantiate_pubsub()
        # No assertion needed, _instantiate_pubsub raises SystemExit on failure

        # --- 5. Service Instantiation (Order Matters!) ---
        # LoggerService needs to be instantiated *before* other services that depend on it.
        # Services are initialized in dependency order
        self._init_strategy_arbitrator()
        self._init_cli_service()

        log.info("Initialization phase complete.")

    async def _start_pubsub_manager(self) -> None:
        """Start the PubSubManager if it exists and has a start method."""
        if self.pubsub and hasattr(self.pubsub, "start"):
            try:
                log.info("Starting PubSubManager...")
                await self.pubsub.start()
                log.info("PubSubManager started.")
            except Exception as e:
                log.exception("FATAL: Failed to start PubSubManager")
                raise PubSubManagerStartFailedExit from e

    async def _create_and_run_service_start_tasks(self) -> list[Any | BaseException]:
        # Changed return type
        """Create and run start tasks for all registered services."""
        log.info("Starting %s services...", len(self.services))
        start_tasks = []
        start_exceptions = []

        for service in self.services:
            service_name = service.__class__.__name__
            if hasattr(service, "start"):
                try:
                    log.debug("Creating start task for %s...", service_name)
                    task = asyncio.create_task(service.start(), name=f"{service_name}_start")
                    start_tasks.append(task)
                    log.info("Start task created for %s.", service_name)
                except Exception as e:
                    log.exception(
                        "Error creating start task for %s",
                        service_name,
                    )
                    start_exceptions.append(f"{service_name}: {e}")  # Store exception for later
            else:
                log.warning("Service %s does not have a start() method.", service_name)

        if start_exceptions:
            # Log collected exceptions from task creation
            log.error("Errors encountered during service task creation: %s", start_exceptions)
            # Potentially raise an error or handle as critical failure if needed

        self.running_tasks.extend(t for t in start_tasks if isinstance(t, asyncio.Task))

        if not self.running_tasks:
            log.warning("No service start tasks were created or all failed at creation.")
            return []  # No tasks to await

        log.info("Waiting for %s service start tasks to complete...", len(self.running_tasks))
        results = await asyncio.gather(*self.running_tasks, return_exceptions=True)
        return list(results)  # Ensure we return a list

    def _handle_service_startup_results(
        self,
        results: list[Any | BaseException],  # Changed parameter type
    ) -> None:
        """Handle the results of service startup tasks."""
        failed_services = []
        for i, result in enumerate(results):
            # Ensure we are within bounds of self.running_tasks if it was modified
            if i < len(self.running_tasks):
                task = self.running_tasks[i]
                task_name = task.get_name() if hasattr(task, "get_name") else f"Task-{i}"
            else:
                # This case should ideally not happen if results and running_tasks are in sync
                task_name = f"Task-{i} (name unknown)"

            if isinstance(result, Exception):
                log.error(
                    "Service task %s failed during startup: %s",
                    task_name,
                    result,
                )
                failed_services.append(task_name)
            else:
                log.info("Service task %s completed startup successfully.", task_name)

        if failed_services:
            log.critical(
                "Critical services failed to start: %s. Initiating shutdown.",
                ", ".join(failed_services),
            )
            shutdown_event.set()
        elif not results and self.services:  # No results but services were expected
            log.warning("No service startup results received, though services exist.")
        else:
            log.info("All services started successfully.")

    async def start(self) -> None:  # Add return type
        """Start all registered services and the PubSub manager."""
        log.info("Starting application services...")
        self.running_tasks = []  # Clear any previous tasks

        await self._start_pubsub_manager()

        # Start ConfigManager file watching if enabled and available
        if self._config_manager_instance and hasattr(
            self._config_manager_instance, "start_watching"
        ):
            log.info("Starting configuration file watcher...")
            self._config_manager_instance.start_watching()

        # Create, run, and get results of service start tasks
        results = await self._create_and_run_service_start_tasks()

        # Handle the results of the service startups
        if results:  # Only handle if there were tasks to run
            self._handle_service_startup_results(results)
        elif not self.services:
            log.info("No services configured to start.")
        else:
            log.warning("No service start tasks were processed.")

        log.info("Application startup sequence complete.")

    async def _initiate_service_shutdown(self) -> list[Exception | Any]:
        """Gathers and executes stop coroutines for services and PubSubManager."""
        log.info("Stopping %s services...", len(self.services))
        stop_coroutines = []
        for service in reversed(self.services):  # Stop in reverse order
            service_name = service.__class__.__name__
            if hasattr(service, "stop"):
                log.debug("Adding stop coroutine for %s...", service_name)
                stop_coroutines.append(service.stop())
            else:
                log.debug("Service %s has no stop() method.", service_name)

        if self.pubsub and hasattr(self.pubsub, "stop"):
            log.debug("Adding stop coroutine for PubSubManager...")
            stop_coroutines.insert(0, self.pubsub.stop())  # Stop PubSub first or last?

        if not stop_coroutines:
            log.info("No services or PubSubManager require explicit stopping.")
            return []

        results = await asyncio.gather(*stop_coroutines, return_exceptions=True)
        for i, result in enumerate(results):
            # Attempt to get service name
            # (this part is a bit tricky as coro might not directly hold it)
            # This is a simplification; robust name retrieval might need passing names along
            coro = stop_coroutines[i]
            instance = getattr(coro, "__self__", None)
            service_name = "UnknownService"
            if instance is self.pubsub:
                service_name = "PubSubManager"
            elif instance:
                service_name = instance.__class__.__name__

            if isinstance(result, Exception):
                log.error("Error stopping service %s: %s", service_name, result)
            else:
                log.debug("Service %s stopped successfully.", service_name)
        return results

    async def _cancel_active_tasks(self) -> None:
        """Cancel all tasks in self.running_tasks."""
        if not self.running_tasks:
            log.info("No active tasks to cancel.")
            return

        log.info("Cancelling %s potentially running service tasks...", len(self.running_tasks))
        for task in self.running_tasks:
            if not task.done():
                task.cancel()

        results = await asyncio.gather(*self.running_tasks, return_exceptions=True)
        cancelled_count = 0
        error_count = 0
        for i, result in enumerate(results):
            task = self.running_tasks[i]
            task_name = task.get_name() if hasattr(task, "get_name") else f"Task-{i}"
            if isinstance(result, asyncio.CancelledError):
                cancelled_count += 1
                log.debug("Task %s cancelled successfully.", task_name)
            elif isinstance(result, Exception):
                error_count += 1
                log.error(
                    "Error during cancellation/completion of task %s: %s",
                    task_name,
                    result,
                )
        log.info(
            "Service task cancellation complete. Cancelled: %s, Errors: %s",
            cancelled_count,
            error_count,
        )
        self.running_tasks.clear()

    async def _shutdown_process_executor(self) -> None:
        """Shuts down the ProcessPoolExecutor."""
        if self.executor:
            log.info("Shutting down ProcessPoolExecutor...")
            try:
                loop = asyncio.get_running_loop()
                # Ensure shutdown is non-blocking in the main async flow
                await loop.run_in_executor(
                    None, functools.partial(self.executor.shutdown, wait=True, cancel_futures=True)
                )
                log.info("ProcessPoolExecutor shut down successfully.")
            except Exception:
                log.exception("Error shutting down ProcessPoolExecutor")
        else:
            log.info("No ProcessPoolExecutor to shut down.")

    async def stop(self) -> None:  # Add return type
        """Stop all registered services, the PubSub manager, and the executor."""
        log.info("Initiating shutdown sequence...")

        # Stop ConfigManager file watching first
        if self._config_manager_instance and hasattr(
            self._config_manager_instance, "stop_watching"
        ):
            log.info("Stopping configuration file watcher...")
            self._config_manager_instance.stop_watching()

        # 1. Stop services and PubSubManager
        await self._initiate_service_shutdown()

        # 2. Cancel any running tasks created during start()
        await self._cancel_active_tasks()

        # 3. Shutdown the executor
        await self._shutdown_process_executor()

        log.info("Shutdown sequence complete.")

    async def run(self) -> None:  # Add return type
        """Run the main application lifecycle: initialize, start, wait, stop."""
        log.info("Running GalFridayApp main lifecycle...")
        # Ensure args is set before calling initialize
        if self.args is None:
            log.error("FATAL: Args not set before calling run(). Exiting.")
            # Or raise an exception
            return  # Or raise RuntimeError("Args not set")

        try:
            # Pass the non-optional args
            await self.initialize(self.args)
            await self.start()
            log.info("Application startup complete. Waiting for shutdown signal...")
            await shutdown_event.wait()  # Wait until shutdown is triggered
        except Exception:  # Remove 'as e' since it's unused
            log.exception("Critical error during application run")
        finally:
            log.info("Shutdown signal received or error encountered. Initiating stop sequence...")
            await self.stop()


# --- Asynchronous Main Function --- #
async def main_async(args: argparse.Namespace) -> None:  # Add return type
    """Set up signal handlers and run the main application loop."""
    # Store the loop in global state for ConfigManager
    global_state.main_event_loop = asyncio.get_running_loop()

    app = GalFridayApp()
    app.args = args  # Set args before running

    loop = asyncio.get_running_loop()

    # Register signal handlers to trigger graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            # Pass only the signal, handle_shutdown sets the global event
            loop.add_signal_handler(sig, functools.partial(handle_shutdown, sig))
            log.info("Registered handler for signal %s", sig.name)
        except NotImplementedError:
            log.warning("Signal handling for %s not supported on this platform.", sig.name)
        except ValueError:
            log.warning("Cannot register signal handler for %s in non-main thread.", sig.name)

    await app.run()
