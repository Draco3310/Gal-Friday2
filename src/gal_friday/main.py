#!/usr/bin/env python
"""
Main entry point for the Gal-Friday trading bot application.

This script initializes all necessary components (configuration, logging, services,
 event bus, executor), wires them together, starts the application, and handles
 graceful shutdown.
"""

import logging
import asyncio
import signal
import os
import sys
import concurrent.futures
import functools
import argparse  # Added for command-line argument parsing
import logging.handlers  # Added for RotatingFileHandler
from typing import List, Optional, Type, Any, TYPE_CHECKING, Union, Tuple, Dict
from decimal import Decimal
from datetime import datetime
import pandas as pd

# --- Conditional Imports for Type Checking --- #
if TYPE_CHECKING:
    from .config_manager import ConfigManager
    from .core.pubsub import PubSubManager
    from .data_ingestor import DataIngestor
    from .feature_engine import FeatureEngine
    from .prediction_service import PredictionService
    from .strategy_arbitrator import StrategyArbitrator
    from .portfolio_manager import PortfolioManager
    from .risk_manager import RiskManager
    from .simulated_execution_handler import SimulatedExecutionHandler
    from .logger_service import LoggerService
    from .monitoring_service import MonitoringService
    from .cli_service import CLIService
    from .market_price_service import MarketPriceService
    from .historical_data_service import HistoricalDataService
    
    # Define a proper protocol/interface for execution handlers
    from typing import Protocol
    
    class ExecutionHandlerProtocol(Protocol):
        """Protocol defining interface for execution handlers."""
        
        def __init__(self, *, 
                     config_manager: 'ConfigManager', 
                     pubsub_manager: 'PubSubManager', 
                     logger_service: 'LoggerService', 
                     **kwargs: Any) -> None: ...
        
        async def start(self) -> None: ...
        
        async def stop(self) -> None: ...
        
        # Add other common methods that execution handlers should implement
        def submit_order(self, order_data: Dict[str, Any]) -> str: ...
        
        def cancel_order(self, order_id: str) -> bool: ...
        
    # Now use this protocol for type annotations
    ExecutionHandlerTypeHint = Type[ExecutionHandlerProtocol]

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
    FeatureEngine = None  # type: ignore[assignment,misc]

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
    LoggerService = None # type: ignore[assignment,misc]

try:
    from .monitoring_service import MonitoringService
except ImportError:
    print("Failed to import MonitoringService")
    MonitoringService = None # type: ignore[assignment,misc]

try:
    from .cli_service import CLIService
except ImportError:
    print("Failed to import CLIService")
    CLIService = None # type: ignore[assignment,misc]

try:
    from .market_price_service import MarketPriceService
except ImportError:
    print("Failed to import MarketPriceService")
    MarketPriceService = None # type: ignore[assignment,misc]

try:
    from .historical_data_service import HistoricalDataService
except ImportError:
    print("Failed to import HistoricalDataService")
    HistoricalDataService = None # type: ignore[assignment,misc]

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
def setup_logging(config: Optional['ConfigManager']) -> None:
    """Configures logging based on the application configuration."""
    # Runtime check still needed
    if config is None or ConfigManager is None:
        log.warning("ConfigManager instance or class not available, cannot configure logging.")
        return

    # No assertion needed here as we checked config is not None
    log_config = config.get("logging", {})
    log_level_name = log_config.get("level", "INFO").upper()
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
        console_handler.setLevel(log_level)  # Handler level defaults to root logger level
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
                # Consider adding jsonlogger to requirements.txt and implementing later.
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
    """Sets the shutdown event when a signal is received."""
    log.warning(f"Received shutdown signal: {sig.name}. Initiating graceful shutdown...")
    shutdown_event.set()


# --- Main Application Class --- #
class GalFridayApp:
    """Encapsulates the main application logic and lifecycle."""

    def __init__(self) -> None: # Add return type
        """Initializes application state attributes."""
        log.info("Initializing GalFridayApp...")
        # Use Optional['ClassName'] string literals for type hints
        self.config: Optional['ConfigManager'] = None
        self.pubsub: Optional['PubSubManager'] = None
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self.services: List[Any] = []  # Use Any for now, can refine later
        self.running_tasks: List[asyncio.Task] = []
        self.args: Optional[argparse.Namespace] = None

        # Store references to specific services after instantiation for DI
        self.logger_service: Optional['LoggerService'] = None
        self.market_price_service: Optional['MarketPriceService'] = None # Added
        self.historical_data_service: Optional['HistoricalDataService'] = None # Added
        self.portfolio_manager: Optional['PortfolioManager'] = None
        # Use the string literal hint defined within TYPE_CHECKING
        self.execution_handler: Union['KrakenExecutionHandler', 'SimulatedExecutionHandler', None] = None
        self.monitoring_service: Optional['MonitoringService'] = None
        self.cli_service: Optional['CLIService'] = None
        self.risk_manager: Optional['RiskManager'] = None
        self.data_ingestor: Optional['DataIngestor'] = None
        self.feature_engine: Optional['FeatureEngine'] = None
        self.prediction_service: Optional['PredictionService'] = None
        self.strategy_arbitrator: Optional['StrategyArbitrator'] = None


    def _load_configuration(self) -> None: # Add return type
        """Loads the application configuration."""
        try:
            if ConfigManager is None: # Runtime check for the class
                raise RuntimeError("ConfigManager class is not available.")
            self.config = ConfigManager(config_path="config/config.yaml")
            log.info("Configuration loaded successfully.")
        except Exception as e:
            log.exception(f"FATAL: Failed to load configuration: {e}", exc_info=True)
            raise SystemExit("Configuration loading failed.")

    def _setup_executor(self) -> None: # Add return type
        """Sets up the ProcessPoolExecutor."""
        if self.config is None or ConfigManager is None: # Runtime checks
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
            log.exception(
                f"ERROR: Failed to create ProcessPoolExecutor: {e}",
                exc_info=True
            )
            self.executor = None

    def _instantiate_pubsub(self) -> None: # Add return type
        """Instantiates the PubSubManager."""
        try:
            if PubSubManager is None: # Runtime check for the class
                raise RuntimeError("PubSubManager class is not available.")
            # Pass the root logger or a specific logger instance
            self.pubsub = PubSubManager(logger=logging.getLogger("gal_friday.pubsub"))
            log.info("PubSubManager instantiated successfully.")
        except Exception as e:
            log.exception(f"FATAL: Failed to instantiate PubSubManager: {e}", exc_info=True)
            raise SystemExit("PubSubManager instantiation failed.")

    def _instantiate_services(self) -> None: # Add return type
        """Instantiates all core services based on configuration and run mode."""
        log.info("Instantiating core services...")
        self.services = []  # Clear any previous list

        # Ensure critical dependencies (instances and classes) exist before proceeding
        if self.args is None:
            raise RuntimeError("Command line arguments not parsed before instantiating services.")
        if self.config is None or ConfigManager is None:
            raise RuntimeError("Configuration not loaded/available before instantiating services.")
        if self.pubsub is None or PubSubManager is None:
            raise RuntimeError("PubSubManager not instantiated/available before instantiating services.")

        # No assertions needed due to checks above

        try:
            # Determine run mode
            run_mode = self.args.mode or self.config.get("trading.mode", "paper")
            log.info(f"Determined run mode: {run_mode}")

            # --- Instantiate services in dependency order --- #

            # 1. LoggerService (depends on PubSub)
            if LoggerService is None:
                 raise RuntimeError("LoggerService class not available.")
            self.logger_service = LoggerService(
                config_manager=self.config,
                pubsub_manager=self.pubsub  # type: ignore[arg-type]
            )
            self.services.append(self.logger_service)
            log.debug("LoggerService instantiated.")

            # 2. MarketPriceService (depends on Config, PubSub, Logger)
            if MarketPriceService is None:
                # Decide if this is fatal or if PortfolioManager can handle None
                log.warning("MarketPriceService class not available. PortfolioManager might fail.")
                self.market_price_service = None
            else:
                # Assuming MarketPriceService is abstract, let's create a concrete implementation
                class ConcreteMarketPriceService(MarketPriceService):
                    def __init__(self, config_manager: 'ConfigManager', 
                                pubsub_manager: 'PubSubManager', 
                                logger_service: 'LoggerService') -> None:
                        self.config = config_manager
                        self.pubsub = pubsub_manager
                        self.logger = logger_service
                        self.logger.info("ConcreteMarketPriceService initialized", source_module=self.__class__.__name__)
                        
                    async def get_latest_price(self, trading_pair: str) -> Optional[Decimal]:
                        self.logger.warning(
                            f"get_latest_price not fully implemented yet for {trading_pair}", 
                            source_module=self.__class__.__name__
                        )
                        return None
                        
                    async def get_bid_ask_spread(self, trading_pair: str) -> Optional[Tuple[Decimal, Decimal]]:
                        self.logger.warning(
                            f"get_bid_ask_spread not fully implemented yet for {trading_pair}", 
                            source_module=self.__class__.__name__
                        )
                        return None
                        
                self.market_price_service = ConcreteMarketPriceService(
                    config_manager=self.config,
                    pubsub_manager=self.pubsub,
                    logger_service=self.logger_service
                )
                self.services.append(self.market_price_service)
                log.debug("MarketPriceService instantiated.")

            # 3. HistoricalDataService (depends on Config, Logger - potentially others)
            if HistoricalDataService is None:
                 # Decide if this is fatal (e.g., for backtesting/simulation) or optional
                 log.warning("HistoricalDataService class not available. SimulatedExecutionHandler might fail.")
                 self.historical_data_service = None
            else:
                 # Assuming HistoricalDataService takes config_manager, logger_service
                 # Create a concrete implementation of HistoricalDataService
                class ConcreteHistoricalDataService(HistoricalDataService):
                    def __init__(self, config_manager: 'ConfigManager', 
                                 logger_service: 'LoggerService') -> None:
                        self.config = config_manager
                        self.logger = logger_service
                        self.logger.info("ConcreteHistoricalDataService initialized", source_module=self.__class__.__name__)
                    
                    def get_next_bar(self, trading_pair: str, timestamp: datetime) -> Optional[pd.Series]:
                        self.logger.warning(
                            f"get_next_bar not implemented yet for {trading_pair}", 
                            source_module=self.__class__.__name__
                        )
                        return None
                        
                    def get_atr(self, trading_pair: str, timestamp: datetime, period: int = 14) -> Optional[Decimal]:
                        self.logger.warning(
                            f"get_atr not implemented yet for {trading_pair}", 
                            source_module=self.__class__.__name__
                        )
                        return None
                    
                    async def get_historical_ohlcv(
                        self, 
                        trading_pair: str, 
                        start_time: datetime, 
                        end_time: datetime, 
                        interval: str
                    ) -> Optional[pd.DataFrame]:
                        self.logger.warning(
                            f"get_historical_ohlcv not implemented yet for {trading_pair}", 
                            source_module=self.__class__.__name__
                        )
                        return None
                        
                    async def get_historical_trades(
                        self, 
                        trading_pair: str, 
                        start_time: datetime, 
                        end_time: datetime, 
                    ) -> Optional[pd.DataFrame]:
                        self.logger.warning(
                            f"get_historical_trades not implemented yet for {trading_pair}", 
                            source_module=self.__class__.__name__
                        )
                        return None
                        
                self.historical_data_service = ConcreteHistoricalDataService(
                    config_manager=self.config,
                    logger_service=self.logger_service
                )
                self.services.append(self.historical_data_service)
                log.debug("HistoricalDataService instantiated.")

            # 4. PortfolioManager (depends on Config, PubSub, MarketPriceService, Logger)
            if PortfolioManager is None:
                raise RuntimeError("PortfolioManager class not available.")
            self.portfolio_manager = PortfolioManager(
                config_manager=self.config,
                pubsub_manager=self.pubsub,
                market_price_service=self.market_price_service, # Pass instance (or None)
                logger_service=self.logger_service
            )
            self.services.append(self.portfolio_manager)
            log.debug("PortfolioManager instantiated.")

            # 5. RiskManager (depends on PubSub, PortfolioManager, Logger)
            if RiskManager is None:
                raise RuntimeError("RiskManager class not available.")
            risk_config = self.config.get_section("risk") if hasattr(self.config, 'get_section') else {}
            self.risk_manager = RiskManager(
                config=risk_config,
                pubsub_manager=self.pubsub,
                portfolio_manager=self.portfolio_manager,
                logger_service=self.logger_service
            )
            self.services.append(self.risk_manager)
            log.debug("RiskManager instantiated.")

            # 6. ExecutionHandler (conditional, depends on Config, PubSub, Logger, potentially HistoricalData)
            self._instantiate_execution_handler(run_mode)
            # _instantiate_execution_handler now appends to self.services
            if self.execution_handler is None:
                 # This should have been raised in the helper function if instantiation failed
                 raise RuntimeError("Execution Handler failed to instantiate.")

            # 7. DataIngestor (depends on Config, PubSub, Logger)
            if DataIngestor is None:
                raise RuntimeError("DataIngestor class not available.")
            self.data_ingestor = DataIngestor(
                config=self.config,
                pubsub_manager=self.pubsub,
                logger_service=self.logger_service
            )
            self.services.append(self.data_ingestor)
            log.debug("DataIngestor instantiated.")

            # 8. FeatureEngine (depends on Config(dict), PubSub, Logger)
            if FeatureEngine is None:
                raise RuntimeError("FeatureEngine class not available.")
            feature_engine_config = self.config.get_all() if hasattr(self.config, 'get_all') else {}
            self.feature_engine = FeatureEngine(
                config=feature_engine_config,
                pubsub_manager=self.pubsub,
                logger_service=self.logger_service
            )
            self.services.append(self.feature_engine)
            log.debug("FeatureEngine instantiated.")

            # 9. PredictionService (depends on Config(dict), PubSub, Logger)
            if PredictionService is None:
                raise RuntimeError("PredictionService class not available.")
            prediction_service_config = self.config.get_section("prediction_service") if hasattr(self.config, 'get_section') else {}
            self.prediction_service = PredictionService(
                config=prediction_service_config,
                pubsub_manager=self.pubsub,
                logger_service=self.logger_service,
                process_pool_executor=self.executor  # type: ignore[arg-type]
            )
            self.services.append(self.prediction_service)
            log.debug("PredictionService instantiated.")

            # 10. StrategyArbitrator (depends on Config(dict), PubSub, Logger)
            if StrategyArbitrator is None:
                raise RuntimeError("StrategyArbitrator class not available.")
            strategy_arbitrator_config = self.config.get_section("strategy_arbitrator") if hasattr(self.config, 'get_section') else {}
            self.strategy_arbitrator = StrategyArbitrator(
                config=strategy_arbitrator_config,
                pubsub_manager=self.pubsub,
                logger_service=self.logger_service
            )
            self.services.append(self.strategy_arbitrator)
            log.debug("StrategyArbitrator instantiated.")

            # 11. MonitoringService (depends on Config, PubSub, PortfolioManager, Logger)
            if MonitoringService is None:
                raise RuntimeError("MonitoringService class not available.")
            self.monitoring_service = MonitoringService(
                config_manager=self.config,
                pubsub_manager=self.pubsub,
                portfolio_manager=self.portfolio_manager,
                logger_service=self.logger_service
            )
            self.services.append(self.monitoring_service)
            log.debug("MonitoringService instantiated.")

            # 12. (Optional) CLIService (depends on MonitoringService, Logger)
            if CLIService is not None:  # Check if the class exists, not if it's truthy
                self.cli_service = CLIService(
                    monitoring_service=self.monitoring_service,
                    logger_service=self.logger_service,
                    main_app_controller=self  # type: ignore[arg-type] # Self is GalFridayApp, not MainAppController
                )
                self.services.append(self.cli_service)
                log.debug("CLIService instantiated.")
            else:
                 self.cli_service = None

            log.info(f"Successfully instantiated {len(self.services)} core services.")

        except Exception as e:
            log.exception(
                f"FATAL: Failed to instantiate services: {e}",
                exc_info=True
            )
            raise SystemExit("Service instantiation failed.")

    def _instantiate_execution_handler(self, run_mode: str) -> None: # Add return type
        """Instantiates the correct ExecutionHandler based on the run mode."""
        self.execution_handler = None

        # Runtime checks for required classes and instances
        if self.config is None or ConfigManager is None:
            raise RuntimeError("Config not loaded/available for ExecutionHandler.")
        if self.pubsub is None or PubSubManager is None:
            raise RuntimeError("PubSub not loaded/available for ExecutionHandler.")
        if self.logger_service is None or LoggerService is None:
             raise RuntimeError("LoggerService not loaded/available for ExecutionHandler.")

        if run_mode == "live":
            if KrakenExecutionHandler is None:
                 raise RuntimeError("KrakenExecutionHandler class not available for live mode.")
            # Assuming KrakenExecutionHandler needs config_manager, pubsub_manager, logger_service
            self.execution_handler = KrakenExecutionHandler(
                config_manager=self.config,
                pubsub_manager=self.pubsub,
                logger_service=self.logger_service
            )
            log.debug("KrakenExecutionHandler instantiated.")

        elif run_mode == "paper":
            if SimulatedExecutionHandler is None:
                 raise RuntimeError("SimulatedExecutionHandler class not available for paper mode.")
            # Check dependency: HistoricalDataService
            if self.historical_data_service is None or HistoricalDataService is None:
                 raise RuntimeError("HistoricalDataService not instantiated/available for SimulatedExecutionHandler.")

            # Assuming SimulatedExecutionHandler needs config, pubsub, data_service, logger
            self.execution_handler = SimulatedExecutionHandler(
                config_manager=self.config,
                pubsub_manager=self.pubsub,
                data_service=self.historical_data_service,  # Use data_service instead of historical_data_service
                logger_service=self.logger_service
            )
            log.debug("SimulatedExecutionHandler instantiated for paper mode.")

        else:
            raise ValueError(f"Unsupported run mode: {run_mode}. Choose 'live' or 'paper'.")

        # Append the instantiated handler to the services list
        if self.execution_handler:
            self.services.append(self.execution_handler)
        else:
             # This path should ideally not be reached due to prior checks/raises
             raise RuntimeError(f"Execution handler instantiation failed unexpectedly for mode: {run_mode}")


    async def initialize(self, args: argparse.Namespace) -> None: # Add return type
        """Loads configuration, sets up logging, and instantiates components."""
        log.info("Initializing application components...")
        self.args = args # args should be guaranteed by main_async

        # --- 1. Configuration Loading ---
        self._load_configuration()
        # No assertion needed, _load_configuration raises SystemExit on failure

        # --- 2. Logging Setup ---
        try:
            # Pass the instance, setup_logging handles None check
            setup_logging(self.config)
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

    async def start(self) -> None: # Add return type
        """Starts all registered services and the PubSub manager."""
        log.info("Starting application services...")
        self.running_tasks = []  # Clear any previous tasks

        # Start PubSubManager first (if it has an async start method)
        if self.pubsub and hasattr(self.pubsub, "start"):
            try:
                log.info("Starting PubSubManager...")
                # PubSub start might not be a task but setup, or could return tasks
                await self.pubsub.start()
                log.info("PubSubManager started.")
            except Exception as e:
                log.exception(f"FATAL: Failed to start PubSubManager: {e}", exc_info=True)
                # Cannot proceed if PubSub fails to start
                raise SystemExit("PubSubManager failed to start.")

        # Start all other services concurrently
        log.info(f"Starting {len(self.services)} services...")
        start_exceptions = []
        for service in self.services:
            service_name = service.__class__.__name__
            if hasattr(service, "start"):
                try:
                    log.debug(f"Creating start task for {service_name}...")
                    task = asyncio.create_task(service.start(), name=f"{service_name}_start")
                    self.running_tasks.append(task)
                    log.info(f"Start task created for {service_name}.")
                except Exception as e:
                    log.exception(
                        f"Error creating start task for {service_name}: {e}", exc_info=True
                    )
                    start_exceptions.append(f"{service_name}: {e}")
            else:
                log.warning(f"Service {service_name} does not have a start() method.")

        if start_exceptions:
            log.error(f"Errors encountered during service task creation: {start_exceptions}")
            # Decide if this is fatal. For now, log and continue, but some services might not run.
            # Consider adding more robust handling, e.g., stopping if critical services fail.

        # Optionally, wait briefly for tasks to start up or check initial health
        await asyncio.sleep(1)  # Small delay to allow tasks to start processing

        log.info("All service start tasks created.")

    async def stop(self) -> None: # Add return type
        """Stops all registered services, the PubSub manager, and the executor."""
        log.info("Initiating shutdown sequence...")

        # 1. Stop services concurrently (reverse order of start is often good practice)
        log.info(f"Stopping {len(self.services)} services...")
        stop_coroutines = []
        for service in reversed(self.services):
            service_name = service.__class__.__name__
            if hasattr(service, "stop"):
                log.debug(f"Adding stop coroutine for {service_name}...")
                stop_coroutines.append(service.stop())
            else:
                log.debug(f"Service {service_name} has no stop() method.")

        # Add PubSub stop if it exists
        if self.pubsub and hasattr(self.pubsub, "stop"):
            log.debug("Adding stop coroutine for PubSubManager...")
            # Add PubSub stop early in the list if other services depend on it during stop
            stop_coroutines.insert(0, self.pubsub.stop())

        if stop_coroutines:
            results = await asyncio.gather(*stop_coroutines, return_exceptions=True)
            # Process results to log specific errors from service stops
            for result, coro in zip(results, stop_coroutines):
                # Attempt to get service name from coroutine (might be fragile)
                # If coro is from pubsub.stop(), handle appropriately
                instance = getattr(coro, "__self__", None)
                if instance is self.pubsub:
                    service_name = "PubSubManager"
                else:
                    service_name = instance.__class__.__name__ if instance else "Unknown"

                if isinstance(result, Exception):
                    log.error(
                        f"Error stopping service {service_name}: {result}",
                        exc_info=result
                    )
                else:
                    log.debug(f"Service {service_name} stopped successfully.")
        log.info("All service stop commands issued.")

        # 2. Cancel any potentially lingering tasks from start (optional but safer)
        # Note: Cancellation might interrupt ongoing operations within services.
        # If services guarantee cleanup in stop(), this might not be strictly necessary,
        # but can help terminate faster if a service hangs.
        # log.info(f"Cancelling {len(self.running_tasks)} running tasks...")
        # for task in self.running_tasks:
        #     if not task.done():
        #         task.cancel()
        # if self.running_tasks:
        #     # Allow tasks to handle cancellation
        #     await asyncio.gather(*self.running_tasks, return_exceptions=True)
        # log.info("Running tasks cancelled.")

        # 3. Shutdown the executor
        if self.executor:
            log.info("Shutting down ProcessPoolExecutor...")
            try:
                self.executor.shutdown(wait=True)  # Wait for tasks to finish
                log.info("ProcessPoolExecutor shut down successfully.")
            except Exception as e:
                log.error(
                    f"Error shutting down ProcessPoolExecutor: {e}",
                    exc_info=True
                )
        else:
            log.info("No ProcessPoolExecutor to shut down.")

        log.info("Shutdown sequence complete.")

    async def run(self) -> None: # Add return type
        """Runs the main application lifecycle: initialize, start, wait, stop."""
        log.info("Running GalFridayApp main lifecycle...")
        # Ensure args is set before calling initialize
        if self.args is None:
             log.error("FATAL: Args not set before calling run(). Exiting.")
             # Or raise an exception
             return # Or raise RuntimeError("Args not set")

        try:
            # Pass the non-optional args
            await self.initialize(self.args)
            await self.start()
            log.info("Application startup complete. Waiting for shutdown signal...")
            await shutdown_event.wait()  # Wait until shutdown is triggered
        except Exception as e:
            log.exception(f"Critical error during application run: {e}", exc_info=True)
        finally:
            log.info("Shutdown signal received or error encountered. Initiating stop sequence...")
            await self.stop()


# --- Asynchronous Main Function --- #
async def main_async(args: argparse.Namespace) -> None: # Add return type
    """Sets up signal handlers and runs the main application loop."""
    app = GalFridayApp()
    app.args = args # Set args before running

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
    """Creates and returns the argument parser for CLI args."""
    parser = argparse.ArgumentParser(description="Gal-Friday Trading Bot")
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
    # Add other potential arguments here (e.g., --config)
    return parser

def main(args: Optional[argparse.Namespace] = None) -> None:
    """Command-line entry point."""
    if args is None:
        args = _parse_arguments()
    
    try:
        asyncio.run(main_async(args))
    except (KeyboardInterrupt, SystemExit):
        # Catch exceptions that signal intentional stop (like SystemExit from failed init)
        log.warning("Application exit triggered.")
    except Exception as e:
        log.exception("Caught unexpected critical error in main execution block", exc_info=True)
    finally:
        log.info("Performing final logging shutdown...")
        logging.shutdown()
        log.info("Gal-Friday Application finished.")

def process_args_and_run() -> None:
    """Main function that processes arguments and runs the application."""
    log.info("Starting Gal-Friday Application...")
    # --- Argument Parsing --- #
    parser = create_arg_parser()
    cli_args = parser.parse_args()

    try:
        asyncio.run(main_async(cli_args))
    except (KeyboardInterrupt, SystemExit):
        # Catch exceptions that signal intentional stop (like SystemExit from failed init)
        log.warning("Application exit triggered.")
    except Exception as e:  # noqa: F841
        log.exception("Caught unexpected critical error in main execution block", exc_info=True)
    finally:
        log.info("Performing final logging shutdown...")
        logging.shutdown()
        log.info("Gal-Friday Application finished.")

if __name__ == "__main__":
    process_args_and_run()
