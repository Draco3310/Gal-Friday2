# File: gal_friday/main.py
# Sections refactored to pass session_maker to _init_strategy_arbitrator and _init_cli_service

# ... (other imports and class definitions from main.py) ...

# --- Conditional Imports for Type Checking --- #
if TYPE_CHECKING:
    # ... (existing TYPE_CHECKING imports) ...
    from sqlalchemy.ext.asyncio import async_sessionmaker # Ensure this is available

    # ... (other existing TYPE_CHECKING imports) ...

# ... (other parts of main.py, including service imports) ...

class GalFridayApp:
    """Encapsulates the main application logic and lifecycle."""

    def __init__(self) -> None:
        """Initialize application state attributes."""
        log.info("Initializing GalFridayApp...")
        # ... (other attributes) ...
        self.config: Optional[ConfigManagerType] = None # Corrected type hint
        self.pubsub: Optional[PubSubManagerType] = None # Corrected type hint
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None # Corrected type hint
        self.services: list[Any] = []
        self.running_tasks: list[asyncio.Task] = []
        self.args: Optional[argparse.Namespace] = None # Corrected type hint

        # Store references to specific services after instantiation for DI
        self.logger_service: Optional[LoggerServiceType] = None # Corrected type hint
        self.db_connection_pool: Optional[DatabaseConnectionPool] = None
        self.session_maker: Optional[async_sessionmaker] = None # async_sessionmaker type from sqlalchemy
        self.migration_manager: Optional[MigrationManager] = None
        self.market_price_service: Optional[MarketPriceServiceType] = None # Corrected type hint
        self.historical_data_service: Optional[HistoricalDataServiceType] = None # Corrected type hint
        self.portfolio_manager: Optional[PortfolioManagerType] = None # Corrected type hint
        self.execution_handler: Optional[_ExecutionHandlerType] = None # Corrected type hint
        self.monitoring_service: Optional[MonitoringServiceType] = None # Corrected type hint
        self.cli_service: Optional[CLIServiceType] = None # Corrected type hint
        self.risk_manager: Optional[RiskManagerType] = None # Corrected type hint
        self.data_ingestor: Optional[DataIngestorType] = None # Corrected type hint
        self.feature_engine: Optional[FeatureEngineType] = None # Corrected type hint
        self.prediction_service: Optional[PredictionServiceType] = None # Corrected type hint
        self.strategy_arbitrator: Optional[StrategyArbitratorType] = None # Corrected type hint
        self.feature_registry_client: Optional[FeatureRegistryClient] = None

        self._config_manager_instance: Optional[ConfigManagerType] = None # Corrected type hint

    # ... (methods like _load_configuration, _setup_executor, _instantiate_pubsub, etc.) ...

    def _init_strategy_arbitrator(self, session_maker: Optional[async_sessionmaker]) -> None: # Added session_maker parameter
        """Instantiate the StrategyArbitrator."""
        self._ensure_strategy_arbitrator_prerequisites()

        if self.config is None:
            raise RuntimeError("Configuration not initialized")
        if self.market_price_service is None: # Added for clarity, though covered by _ensure
            self._raise_dependency_not_instantiated("StrategyArbitrator", "MarketPriceService")
            return # Should not be reached if _ensure works
        if self.pubsub is None: # Added for clarity
            self._raise_dependency_not_instantiated("StrategyArbitrator", "PubSubManager")
            return
        if self.logger_service is None: # Added for clarity
            self._raise_dependency_not_instantiated("StrategyArbitrator", "LoggerService")
            return


        strategy_arbitrator_config = self.config.get("strategy_arbitrator", {})

        if FeatureRegistryClient is not None:
            self.feature_registry_client = FeatureRegistryClient()
            log.debug("FeatureRegistryClient instantiated.")
        else:
            log.warning("FeatureRegistryClient not available, StrategyArbitrator will operate without feature validation.")
            self.feature_registry_client = None

        # Assuming StrategyArbitrator constructor is updated to accept session_maker
        self.strategy_arbitrator = StrategyArbitrator(
            config=strategy_arbitrator_config,
            pubsub_manager=self.pubsub,
            logger_service=self.logger_service,
            market_price_service=self.market_price_service,
            feature_registry_client=self.feature_registry_client,
            session_maker=session_maker # Pass session_maker
        )
        log.debug("StrategyArbitrator instantiated.")
        # Add to services list if it has start/stop methods and needs to be managed
        if hasattr(self.strategy_arbitrator, "start") and hasattr(self.strategy_arbitrator, "stop"):
             self.services.append(self.strategy_arbitrator)


    def _init_cli_service(self, session_maker: Optional[async_sessionmaker]) -> None: # Added session_maker parameter
        """Instantiate the CLIService."""
        if CLIService is not None:
            if self.monitoring_service is None:
                self._raise_dependency_not_instantiated("CLIService", "MonitoringService instance")
                return # Should not be reached
            if self.portfolio_manager is None:
                self._raise_dependency_not_instantiated("CLIService", "PortfolioManager instance")
                return # Should not be reached
            if self.logger_service is None:
                self._raise_dependency_not_instantiated("CLIService", "LoggerService")
                return # Should not be reached


            # Assuming CLIService constructor is updated to accept session_maker
            self.cli_service = CLIService(
                monitoring_service=self.monitoring_service,
                logger_service=self.logger_service,
                main_app_controller=self,
                portfolio_manager=self.portfolio_manager,
                session_maker=session_maker # Pass session_maker
            )
            self.services.append(self.cli_service)
            log.debug("CLIService instantiated.")
        else:
            self.cli_service = None
            log.info("CLIService not available or not configured.")

    # ... (other _ensure_ and _raise_ methods) ...

    async def initialize(self, args: argparse.Namespace) -> None:
        """Load configuration, set up logging, and instantiate components."""
        log.info("Initializing GalFridayApp (Version: %s)...", __version__)
        self.args = args

        # --- 1. Configuration Loading ---
        self._load_configuration(args.config)

        # --- 2. Logging Setup ---
        # ... (logging setup logic) ...

        # --- 3. Executor Setup ---
        self._setup_executor()

        # --- 4. PubSub Manager Instantiation ---
        self._instantiate_pubsub()

        # --- 5. Database Connection Pool and Session Maker ---
        if DatabaseConnectionPool is not None and self.config is not None:
            temp_db_logger = logging.getLogger("gal_friday.db_pool_init")
            self.db_connection_pool = DatabaseConnectionPool(
                config=self.config,
                logger=temp_db_logger, # type: ignore
            )
            await self.db_connection_pool.initialize()
            self.session_maker = self.db_connection_pool.get_session_maker()
            if not self.session_maker:
                log.critical("Failed to get session_maker from DatabaseConnectionPool. DB-dependent services will fail.")
                raise DependencyMissingError("Application", "session_maker from DatabaseConnectionPool")
            log.info("DatabaseConnectionPool initialized and session_maker created.")
        else:
            log.critical("DatabaseConnectionPool or its dependencies (ConfigManager) are missing.")
            raise DependencyMissingError("Application", "DatabaseConnectionPool or ConfigManager")

        # --- 6. LoggerService Full Instantiation (with DB capabilities) ---
        if LoggerService is None:
            self._raise_logger_service_instantiation_failed()
            return # Should not be reached

        try:
            self.logger_service = LoggerService(
                config_manager=self.config, # type: ignore
                pubsub_manager=self.pubsub, # type: ignore
                db_session_maker=self.session_maker,
            )
            self.services.append(self.logger_service)
            log.info("LoggerService instantiated/configured with DB support.")
        except Exception as e:
            log.exception("FATAL: Failed to instantiate full LoggerService")
            raise LoggerServiceInstantiationFailedExit from e

        # --- 7. MigrationManager Setup ---
        # ... (migration manager logic, ensuring it uses self.logger_service and self.session_maker if needed indirectly via config or direct pass) ...
        if MigrationManager is not None and self.logger_service is not None:
            self.migration_manager = MigrationManager(
                logger=self.logger_service,
                project_root_path="/app", # Or dynamically determine
                # If MigrationManager needs direct db access, it might need session_maker or engine
                # For now, assuming it uses config or gets engine from a shared place if needed.
            )
            log.info("MigrationManager instantiated.")
            try:
                log.info("Running database migrations to head...")
                await asyncio.to_thread(self.migration_manager.upgrade_to_head) # Ensure this is how it's run
                log.info("Database migrations completed.")
            except Exception:
                log.exception("Failed to run database migrations.")
                raise
        else:
            log.critical("MigrationManager or LoggerService missing, cannot run migrations.")
            raise DependencyMissingError("Application", "MigrationManager or LoggerService")


        # --- 8. Other Service Instantiation (Order Matters!) ---
        # Instantiate services that might need session_maker.
        # The original code has a comment here about refactoring.
        # We are now performing that refactoring.

        # Example of initializing other services that might need session_maker:
        # if PortfolioManager is not None and self.market_price_service is not None and self.pubsub is not None and self.logger_service is not None:
        #     self.portfolio_manager = PortfolioManager(
        #         config_manager=self.config,
        #         market_price_service=self.market_price_service,
        #         pubsub_manager=self.pubsub,
        #         logger_service=self.logger_service,
        #         session_maker=self.session_maker # Pass session_maker
        #     )
        #     self.services.append(self.portfolio_manager)
        #     log.debug("PortfolioManager instantiated with session_maker.")
        # else:
        #     # Handle missing dependencies for PortfolioManager
        #     self._raise_dependency_not_instantiated("Application", "PortfolioManager or its dependencies")


        # Call _init_strategy_arbitrator and _init_cli_service with session_maker
        self._init_strategy_arbitrator(self.session_maker) # Pass self.session_maker
        self._init_cli_service(self.session_maker)         # Pass self.session_maker

        # ... (Initialize other services like RiskManager, DataIngestor, FeatureEngine, PredictionService, MonitoringService)
        # These should also be checked if they need session_maker and refactored similarly if so.
        # For example, RiskManager might need it:
        # if RiskManager is not None and self.market_price_service is not None and self.portfolio_manager is not None and self.pubsub is not None and self.logger_service is not None:
        #    self.risk_manager = RiskManager(
        #        config_manager=self.config,
        #        market_price_service=self.market_price_service,
        #        portfolio_manager=self.portfolio_manager,
        #        pubsub_manager=self.pubsub,
        #        logger_service=self.logger_service,
        #        session_maker=self.session_maker # Pass session_maker
        #    )
        #    self.services.append(self.risk_manager)
        #    log.debug("RiskManager instantiated with session_maker.")


        log.info("Initialization phase complete.")

    # ... (rest of GalFridayApp class: _start_pubsub_manager, start, stop, run, etc.) ...

# ... (main_async function and script entry point) ...