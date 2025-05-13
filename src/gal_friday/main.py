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
import os
import signal
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import pandas as pd

# Version information
__version__ = "0.1.0"  # Add version tracking

# --- Conditional Imports for Type Checking --- #
if TYPE_CHECKING:
    # from .simulated_market_price_service import SimulatedMarketPriceService # Removed for F811
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
            **kwargs: Any,
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
        def submit_order(self, order_data: Dict[str, Any]) -> str:
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
    ExecutionHandlerTypeHint = Type[ExecutionHandlerProtocol]
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
    from .feature_engine import FeatureEngine
except ImportError:
    print("Failed to import FeatureEngine")
    FeatureEngine = None

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
    from .execution.kraken import KrakenExecutionHandler  # noqa: F811
except ImportError as e:
    print(f"Failed to import KrakenExecutionHandler: {e}")
    KrakenExecutionHandler = None  # type: ignore

try:
    from .simulated_execution_handler import SimulatedExecutionHandler  # noqa: F811
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


# --- Logging Setup Function --- #
# Use string literal for the type hint
def setup_logging(
    config: Optional["ConfigManagerType"], log_level_override: Optional[str] = None
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

    log.info(f"Root logger level set to {log_level_name}")

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
            log_dir = os.path.dirname(log_filename)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir)
                    log.info(f"Created log directory: {log_dir}")
                except OSError as e:
                    log.error(f"Could not create log directory {log_dir}: {e}", exc_info=True)
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
                log.info(f"File logging enabled: {log_filename}")
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
    log.warning(f"Received shutdown signal: {sig.name}. Initiating graceful shutdown...")
    shutdown_event.set()


# --- Main Application Class --- #
class GalFridayApp:
    """Encapsulates the main application logic and lifecycle."""

    def __init__(self) -> None:  # Add return type
        """Initialize application state attributes."""
        log.info("Initializing GalFridayApp...")
        # Use Optional['ClassName'] string literals for type hints
        self.config: Optional["ConfigManagerType"] = None
        self.pubsub: Optional["PubSubManagerType"] = None
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self.services: List[Any] = []  # Use Any for now, can refine later
        self.running_tasks: List[asyncio.Task] = []
        self.args: Optional[argparse.Namespace] = None

        # Store references to specific services after instantiation for DI
        self.logger_service: Optional["LoggerServiceType"] = None
        # Added
        self.market_price_service: Optional["MarketPriceServiceType"] = None
        # Added
        self.historical_data_service: Optional["HistoricalDataServiceType"] = None
        self.portfolio_manager: Optional["PortfolioManagerType"] = None
        # Use the type alias defined in TYPE_CHECKING
        self.execution_handler: Optional[_ExecutionHandlerType] = None
        self.monitoring_service: Optional["MonitoringServiceType"] = None
        self.cli_service: Optional["CLIServiceType"] = None
        self.risk_manager: Optional["RiskManagerType"] = None
        self.data_ingestor: Optional["DataIngestorType"] = None
        self.feature_engine: Optional["FeatureEngineType"] = None
        self.prediction_service: Optional["PredictionServiceType"] = None
        self.strategy_arbitrator: Optional["StrategyArbitratorType"] = None

    def _load_configuration(self, config_path: str) -> None:  # Accept config_path parameter
        """Load the application configuration."""
        try:
            if ConfigManager is None:  # Runtime check for the class
                raise RuntimeError("ConfigManager class is not available.")
            self.config = ConfigManager(config_path=config_path)  # Use the provided path
            log.info(f"Configuration loaded successfully from: {config_path}")
        except Exception as e:
            log.exception(
                f"FATAL: Failed to load configuration from {config_path}: {e}", exc_info=True
            )
            raise SystemExit("Configuration loading failed.")

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
                log.warning(f"Invalid executor_workers count ({max_workers}), defaulting to 1.")
                max_workers = 1
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
            log.info(f"ProcessPoolExecutor created with max_workers={max_workers}.")
        except Exception as e:
            log.exception(f"ERROR: Failed to create ProcessPoolExecutor: {e}", exc_info=True)
            self.executor = None

    def _instantiate_pubsub(self) -> None:  # Add return type
        """Instantiate the PubSubManager."""
        try:
            if PubSubManager is None:  # Runtime check for the class
                raise RuntimeError("PubSubManager class is not available.")
            if self.config is None:
                raise RuntimeError("Configuration not loaded before instantiating PubSubManager.")
            # Pass the root logger or a specific logger instance
            self.pubsub = PubSubManager(
                logger=logging.getLogger("gal_friday.pubsub"),
                config_manager=self.config,  # Added config_manager
            )
            log.info("PubSubManager instantiated successfully.")
        except Exception as e:
            log.exception(f"FATAL: Failed to instantiate PubSubManager: {e}", exc_info=True)
            raise SystemExit("PubSubManager instantiation failed.")

    def _instantiate_services(self) -> None:  # noqa: C901
        """Instantiate all core services based on configuration and run mode."""
        log.info("Instantiating core services...")
        self.services = []  # Clear any previous list

        if self.args is None:
            raise RuntimeError("Command line arguments not parsed before instantiating services.")
        if self.config is None or ConfigManager is None:
            raise RuntimeError("Configuration not loaded/available before instantiating services.")
        if self.pubsub is None or PubSubManager is None:
            raise RuntimeError(
                "PubSubManager not instantiated/available before instantiating services."
            )

        run_mode = self.args.mode or self.config.get("trading.mode", "paper")
        log.info(f"Determined run mode: {run_mode}")

        try:
            # 1. LoggerService
            if LoggerService is None:
                raise RuntimeError("LoggerService class not available.")
            self.logger_service = LoggerService(
                config_manager=self.config, pubsub_manager=self.pubsub
            )
            self.services.append(self.logger_service)
            log.debug("LoggerService instantiated.")
            if self.logger_service is None:
                raise SystemExit("LoggerService instantiation failed.")

            # 2. HistoricalDataService
            # Assumes KrakenHistoricalDataService is the
            # default for now if HistoricalDataService is needed
            # This might need to be configurable or mode-dependent
            if HistoricalDataService is None:
                log.warning("HistoricalDataService class not available.")
                self.historical_data_service = None
            elif KrakenHistoricalDataService is not None:
                historical_data_config = (
                    self.config.get_section("historical_data")
                    if hasattr(self.config, "get_section")
                    else {}
                )
                self.historical_data_service = KrakenHistoricalDataService(
                    config=historical_data_config, logger_service=self.logger_service
                )
                self.services.append(self.historical_data_service)
                log.debug("KrakenHistoricalDataService instantiated.")
            else:
                self.historical_data_service = None
                log.warning("No concrete HistoricalDataService implementation found.")

            # 3. MarketPriceService
            if MarketPriceService is None:
                raise RuntimeError(
                    "MarketPriceService class not available. This is a critical service."
                )

            if run_mode == "live":
                if KrakenMarketPriceService is None:
                    raise RuntimeError("KrakenMarketPriceService not available for live mode.")
                self.market_price_service = KrakenMarketPriceService(
                    config_manager=self.config, logger_service=self.logger_service
                )
                log.debug("KrakenMarketPriceService instantiated for live mode.")
            elif (
                SimulatedMarketPriceService is not None
            ):  # For paper or backtest (if main handles backtest init)
                # TODO: historical_data dict for
                # SimulatedMarketPriceService needs to be properly populated,
                # perhaps from HistoricalDataService or specific backtest setup.
                # For now, using empty dict to allow instantiation.
                sim_hist_data: Dict[str, pd.DataFrame] = {}
                # Assuming LoggerService has a .logger attribute that is a logging.Logger
                sim_logger = (
                    self.logger_service.logger
                    if hasattr(self.logger_service, "logger")
                    else logging.getLogger("SimulatedMarketPrice")
                )

                self.market_price_service = SimulatedMarketPriceService(
                    historical_data=sim_hist_data, config_manager=self.config, logger=sim_logger
                )
                log.debug("SimulatedMarketPriceService instantiated for non-live mode.")
            else:
                raise RuntimeError(
                    f"Cannot instantiate MarketPriceService for mode '{run_mode}'. "
                    f"Missing implementation."
                )

            if self.market_price_service is None:
                raise SystemExit("MarketPriceService instantiation failed critically.")
            self.services.append(self.market_price_service)

            # 4. PortfolioManager
            if self.market_price_service is None:
                raise RuntimeError(
                    "MarketPriceService not instantiated. PortfolioManager cannot be created."
                )
            if PortfolioManager is None:
                raise RuntimeError("PortfolioManager class not available.")
            self.portfolio_manager = PortfolioManager(
                config_manager=self.config,
                pubsub_manager=self.pubsub,
                market_price_service=self.market_price_service,
                logger_service=self.logger_service,
            )
            self.services.append(self.portfolio_manager)
            log.debug("PortfolioManager instantiated.")
            if self.portfolio_manager is None:
                raise SystemExit("PortfolioManager instantiation failed.")

            # 5. RiskManager
            if self.market_price_service is None:
                raise RuntimeError(
                    "MarketPriceService not instantiated. RiskManager cannot be created."
                )
            if RiskManager is None:
                raise RuntimeError("RiskManager class not available.")
            risk_config = self.config.get("risk") if hasattr(self.config, "get") else {}
            self.risk_manager = RiskManager(
                config=risk_config,
                pubsub_manager=self.pubsub,
                portfolio_manager=self.portfolio_manager,
                logger_service=self.logger_service,
                market_price_service=self.market_price_service,
            )
            self.services.append(self.risk_manager)
            log.debug("RiskManager instantiated.")
            if self.risk_manager is None:
                raise SystemExit("RiskManager instantiation failed.")

            # 6. MonitoringService
            if MonitoringService is None:
                raise RuntimeError("MonitoringService class not available.")
            self.monitoring_service = MonitoringService(
                config_manager=self.config,
                pubsub_manager=self.pubsub,
                portfolio_manager=self.portfolio_manager,
                logger_service=self.logger_service,
            )
            self.services.append(self.monitoring_service)
            log.debug("MonitoringService instantiated.")

            # 7. ExecutionHandler
            self._instantiate_execution_handler(run_mode)
            if self.execution_handler is None:
                raise SystemExit(f"Execution Handler failed to instantiate for mode: {run_mode}")

            # 8. DataIngestor
            if DataIngestor is None:
                raise RuntimeError("DataIngestor class not available.")
            self.data_ingestor = DataIngestor(
                config=self.config, pubsub_manager=self.pubsub, logger_service=self.logger_service
            )
            self.services.append(self.data_ingestor)
            log.debug("DataIngestor instantiated.")

            # 9. FeatureEngine
            if FeatureEngine is None:
                raise RuntimeError("FeatureEngine class not available.")
            feature_engine_config = (
                self.config.get_all() if hasattr(self.config, "get_all") else {}
            )
            # FeatureEngine might need historical_data_service for initial data loading
            self.feature_engine = FeatureEngine(
                config=feature_engine_config,
                pubsub_manager=self.pubsub,
                logger_service=self.logger_service,
                historical_data_service=self.historical_data_service,  # Pass HDS
            )
            self.services.append(self.feature_engine)
            log.debug("FeatureEngine instantiated.")

            # 10. PredictionService
            if PredictionService is None:
                raise RuntimeError("PredictionService class not available.")
            prediction_service_config = (
                self.config.get_section("prediction_service")
                if hasattr(self.config, "get_section")
                else {}
            )
            self.prediction_service = PredictionService(
                config=prediction_service_config,
                pubsub_manager=self.pubsub,
                logger_service=self.logger_service,
                process_pool_executor=self.executor,  # type: ignore[arg-type]
            )
            self.services.append(self.prediction_service)
            log.debug("PredictionService instantiated.")

            # 11. StrategyArbitrator
            if self.market_price_service is None:
                raise RuntimeError(
                    "MarketPriceService not instantiated. StrategyArbitrator cannot be created."
                )
            if StrategyArbitrator is None:
                raise RuntimeError("StrategyArbitrator class not available.")
            strategy_arbitrator_config = (
                self.config.get("strategy_arbitrator") if hasattr(self.config, "get") else {}
            )
            self.strategy_arbitrator = StrategyArbitrator(
                config=strategy_arbitrator_config,
                pubsub_manager=self.pubsub,
                logger_service=self.logger_service,
                market_price_service=self.market_price_service,
            )
            self.services.append(self.strategy_arbitrator)
            log.debug("StrategyArbitrator instantiated.")

            # 12. CLIService
            if CLIService is not None:
                self.cli_service = CLIService(
                    monitoring_service=self.monitoring_service,
                    logger_service=self.logger_service,
                    main_app_controller=self,
                    portfolio_manager=self.portfolio_manager,  # Pass portfolio_manager to CLI
                )
                self.services.append(self.cli_service)
                log.debug("CLIService instantiated.")
            else:
                self.cli_service = None
                log.info("CLIService not available or not configured.")

            log.info(f"Successfully instantiated {len(self.services)} core services.")

        except Exception as e:
            log.exception(f"FATAL: Failed to instantiate services: {e}", exc_info=True)
            raise SystemExit("Service instantiation failed.")

    # Add return type
    def _instantiate_execution_handler(self, run_mode: str) -> None:
        """Instantiate the correct ExecutionHandler based on the run mode."""
        self.execution_handler = None

        # Runtime checks for required classes and instances
        if self.config is None or ConfigManager is None:
            raise RuntimeError("Config not loaded/available for ExecutionHandler.")
        if self.pubsub is None or PubSubManager is None:
            raise RuntimeError("PubSub not loaded/available for ExecutionHandler.")
        if self.logger_service is None or LoggerService is None:
            raise RuntimeError("LoggerService not loaded/available for ExecutionHandler.")
        if self.monitoring_service is None or MonitoringService is None:  # Added check
            raise RuntimeError("MonitoringService not loaded/available for ExecutionHandler.")

        if run_mode == "live":
            if KrakenExecutionHandler is None:
                raise RuntimeError("KrakenExecutionHandler class not available for live mode.")
            # Assuming KrakenExecutionHandler needs config_manager,
            # pubsub_manager, logger_service
            self.execution_handler = KrakenExecutionHandler(
                config_manager=self.config,
                pubsub_manager=self.pubsub,
                logger_service=self.logger_service,
                monitoring_service=self.monitoring_service,
            )
            log.debug("KrakenExecutionHandler instantiated.")

        elif run_mode == "paper":
            if SimulatedExecutionHandler is None:
                raise RuntimeError("SimulatedExecutionHandler class not available for paper mode.")
            # Check dependency: HistoricalDataService
            if self.historical_data_service is None or HistoricalDataService is None:
                raise RuntimeError(
                    "HistoricalDataService not instantiated/available for "
                    "SimulatedExecutionHandler."
                )
            # Removed the specific check for monitoring_service for
            # SimulatedExecutionHandler as it's not a dependency.
            # The general check at the start of the function handles cases where
            # MonitoringService might be needed by other handlers.

            # Assuming SimulatedExecutionHandler needs config, pubsub,
            # data_service, logger
            self.execution_handler = SimulatedExecutionHandler(
                config_manager=self.config,
                pubsub_manager=self.pubsub,
                data_service=self.historical_data_service,
                logger_service=self.logger_service,
            )
            log.debug("SimulatedExecutionHandler instantiated for paper mode.")

        else:
            raise ValueError(f"Unsupported run mode: {run_mode}. Choose 'live' or 'paper'.")

        # Append the instantiated handler to the services list
        if self.execution_handler:
            self.services.append(self.execution_handler)
        else:
            # This path should ideally not be reached due to prior
            # checks/raises
            raise RuntimeError(
                f"Execution handler instantiation failed unexpectedly for mode: {run_mode}"
            )

    async def initialize(self, args: argparse.Namespace) -> None:  # Add return type
        """Load configuration, set up logging, and instantiate components."""
        log.info(f"Initializing GalFridayApp (Version: {__version__})...")
        self.args = args  # args should be guaranteed by main_async

        # --- 1. Configuration Loading ---
        self._load_configuration(args.config)
        # No assertion needed, _load_configuration raises SystemExit on failure

        # --- 2. Logging Setup ---
        try:
            # Pass the instance and log_level override
            setup_logging(self.config, args.log_level)
            log.info("Logging configured successfully.")
        except Exception as e:
            log.exception(f"ERROR: Failed to configure logging: {e}", exc_info=True)
            # Continue running with basic logging if setup fails

        # --- 3. Executor Setup ---
        self._setup_executor()

        # --- 4. PubSub Manager Instantiation ---
        self._instantiate_pubsub()
        # No assertion needed, _instantiate_pubsub raises SystemExit on failure

        # --- 5. Service Instantiation (Order Matters!) ---
        # LoggerService needs to be instantiated *before* other services that depend on it.
        # Rearranging instantiation order might be necessary based on dependencies.
        # The current _instantiate_services attempts this, but dependencies like
        # MarketPriceService and HistoricalDataService need proper handling.
        self._instantiate_services()

        log.info("Initialization phase complete.")

    async def _start_pubsub_manager(self) -> None:
        """Start the PubSubManager if it exists and has a start method."""
        if self.pubsub and hasattr(self.pubsub, "start"):
            try:
                log.info("Starting PubSubManager...")
                await self.pubsub.start()
                log.info("PubSubManager started.")
            except Exception as e:
                log.exception(f"FATAL: Failed to start PubSubManager: {e}", exc_info=True)
                raise SystemExit("PubSubManager failed to start.")

    async def _create_and_run_service_start_tasks(self) -> List[Union[Any, BaseException]]:
        # Changed return type
        """Create and run start tasks for all registered services."""
        log.info(f"Starting {len(self.services)} services...")
        start_tasks = []
        start_exceptions = []

        for service in self.services:
            service_name = service.__class__.__name__
            if hasattr(service, "start"):
                try:
                    log.debug(f"Creating start task for {service_name}...")
                    task = asyncio.create_task(service.start(), name=f"{service_name}_start")
                    start_tasks.append(task)
                    log.info(f"Start task created for {service_name}.")
                except Exception as e:
                    log.exception(
                        f"Error creating start task for {service_name}: {e}", exc_info=True
                    )
                    start_exceptions.append(f"{service_name}: {e}")  # Store exception for later
            else:
                log.warning(f"Service {service_name} does not have a start() method.")

        if start_exceptions:
            # Log collected exceptions from task creation
            log.error(f"Errors encountered during service task creation: {start_exceptions}")
            # Potentially raise an error or handle as critical failure if needed

        self.running_tasks.extend(t for t in start_tasks if isinstance(t, asyncio.Task))

        if not self.running_tasks:
            log.warning("No service start tasks were created or all failed at creation.")
            return []  # No tasks to await

        log.info(f"Waiting for {len(self.running_tasks)} service start tasks to complete...")
        results = await asyncio.gather(*self.running_tasks, return_exceptions=True)
        return results

    def _handle_service_startup_results(
        self, results: List[Union[Any, BaseException]]  # Changed parameter type
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
                    f"Service task {task_name} failed during startup: {result}",
                    exc_info=result if not isinstance(result, asyncio.CancelledError) else None,
                )
                failed_services.append(task_name)
            else:
                log.info(f"Service task {task_name} completed startup successfully.")

        if failed_services:
            log.critical(
                f"Critical services failed to start: {', '.join(failed_services)}. "
                "Initiating shutdown."
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

    async def _initiate_service_shutdown(self) -> List[Union[Exception, Any]]:
        """Gathers and executes stop coroutines for services and PubSubManager."""
        log.info(f"Stopping {len(self.services)} services...")
        stop_coroutines = []
        for service in reversed(self.services):  # Stop in reverse order
            service_name = service.__class__.__name__
            if hasattr(service, "stop"):
                log.debug(f"Adding stop coroutine for {service_name}...")
                stop_coroutines.append(service.stop())
            else:
                log.debug(f"Service {service_name} has no stop() method.")

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
                log.error(f"Error stopping service {service_name}: {result}", exc_info=result)
            else:
                log.debug(f"Service {service_name} stopped successfully.")
        return results

    async def _cancel_active_tasks(self) -> None:
        """Cancel all tasks in self.running_tasks."""
        if not self.running_tasks:
            log.info("No active tasks to cancel.")
            return

        log.info(f"Cancelling {len(self.running_tasks)} potentially running service tasks...")
        for task in self.running_tasks:
            if not task.done():
                task.cancel()

        results = await asyncio.gather(*self.running_tasks, return_exceptions=True)
        cancelled_count = 0
        error_count = 0
        for i, result in enumerate(results):
            task = self.running_tasks[i]
            if hasattr(task, "get_name"):
                task_name = task.get_name()
            else:
                task_name = f"Task-{i}"
            if isinstance(result, asyncio.CancelledError):
                cancelled_count += 1
                log.debug(f"Task {task_name} cancelled successfully.")
            elif isinstance(result, Exception):
                error_count += 1
                log.error(
                    f"Error during cancellation/completion of task {task_name}: {result}",
                    exc_info=result,
                )
        log.info(
            f"Service task cancellation complete. Cancelled: {cancelled_count}, "
            f"Errors: {error_count}"
        )
        self.running_tasks.clear()

    async def _shutdown_process_executor(self) -> None:
        """Shuts down the ProcessPoolExecutor."""
        if self.executor:
            log.info("Shutting down ProcessPoolExecutor...")
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.executor.shutdown, True)  # wait=True
                log.info("ProcessPoolExecutor shut down successfully.")
            except Exception as e:
                log.error(f"Error shutting down ProcessPoolExecutor: {e}", exc_info=True)
        else:
            log.info("No ProcessPoolExecutor to shut down.")

    async def stop(self) -> None:  # Add return type
        """Stop all registered services, the PubSub manager, and the executor."""
        log.info("Initiating shutdown sequence...")

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
    app = GalFridayApp()
    app.args = args  # Set args before running

    loop = asyncio.get_running_loop()

    # Register signal handlers to trigger graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            # Pass only the signal, handle_shutdown sets the global event
            loop.add_signal_handler(sig, functools.partial(handle_shutdown, sig))
            log.info(f"Registered handler for signal {sig.name}")
        except NotImplementedError:
            log.warning(f"Signal handling for {sig.name} not supported on this platform.")
        except ValueError:
            log.warning(f"Cannot register signal handler for {sig.name} in non-main thread.")

    await app.run()


# --- Main Application Logic (Entry Point) --- #


def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Gal-Friday Trading Bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "paper"],
        default=None,
        help=(
            "Trading mode: 'live' for real trading, 'paper' for simulated "
            "trading with live data."
        ),
    )
    # Add other potential arguments here (e.g., --config)
    return parser.parse_args()


def create_arg_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for CLI args."""
    parser = argparse.ArgumentParser(
        description=f"Gal-Friday Trading Bot (Version: {__version__})"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "paper"],
        default=None,  # Default to None, will fallback to config value if not provided
        help=(
            "Trading mode: 'live' for real trading, 'paper' for simulated "
            "trading with live data."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",  # Default config path
        help="Path to the main configuration file (YAML format).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Override log level specified in config file.",
    )
    return parser


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Execute the application with the given command-line arguments.

    Args
    ----
        args: Command-line arguments (parsed). If None, arguments will be parsed.
    """
    if args is None:
        args = _parse_arguments()

    try:
        asyncio.run(main_async(args))
    except (KeyboardInterrupt, SystemExit):
        # Catch exceptions that signal intentional stop (like SystemExit from
        # failed init)
        log.warning("Application exit triggered.")
    except Exception:  # Remove 'as e'
        log.exception("Caught unexpected critical error in main execution block", exc_info=True)
    finally:
        log.info("Performing final logging shutdown...")
        logging.shutdown()
        log.info("Gal-Friday Application finished.")


def process_args_and_run() -> None:
    """Process command line arguments and run the application.

    This function is the main entry point when running the module directly.
    It sets up argument parsing and runs the application with proper error handling.
    """
    log.info(f"--- Starting Gal-Friday Application (Version: {__version__}) ---")
    # --- Argument Parsing --- #
    parser = create_arg_parser()
    cli_args = parser.parse_args()

    try:
        main(cli_args)
    except (KeyboardInterrupt, SystemExit):
        # Catch exceptions that signal intentional stop (like SystemExit from
        # failed init)
        log.warning("Application exit triggered.")
    except Exception:  # Remove the 'as e' part
        log.exception("Caught unexpected critical error in main execution block", exc_info=True)
    finally:
        log.info("Performing final logging shutdown...")
        logging.shutdown()
        log.info("Gal-Friday Application finished.")


if __name__ == "__main__":
    process_args_and_run()
