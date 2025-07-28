#!/usr/bin/env python
"""Main entry point for the Gal-Friday trading bot application.

This script initializes all necessary components (configuration, logging, services,
 event bus, executor), wires them together, starts the application, and handles
 graceful shutdown.
"""

import argparse  # Added for command-line argument parsing
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import functools
import logging
import logging.handlers  # Added for RotatingFileHandler
import os
from pathlib import Path  # PTHxxx fix: Import Path
import signal
import sys
import traceback
from typing import TYPE_CHECKING, Any, Optional, cast

import asyncio
import concurrent.futures
from pydantic import BaseModel, Field, field_validator
from pythonjsonlogger import jsonlogger

from .cli_service import CLIService as CLIServiceType

# Import exchange specification classes
from .config_manager import ConfigManager as ConfigManagerType
from .core.application_services import create_application_container
from .core.asset_registry import ExchangeSpecification, ExchangeType
from .core.dependency_validation import DependencyValidator
from .core.service_container import ServiceContainer
from .core.service_initialization_builder import (
    ServiceInitializationDirector,
)
from .core.type_guards import (
    DependencyTypeGuards,
    TypeSafeAssertion,
    ensure_config_manager,
    ensure_not_none,
    ensure_service_container,
)

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
from .services.service_orchestrator import ServiceOrchestrator
from .strategy_arbitrator import StrategyArbitrator as StrategyArbitratorType

# Version information
__version__ = "0.1.0"  # Add version tracking

# Error messages
_CONFIG_NOT_INITIALIZED_MSG = "ConfigManager is not initialized"


# --- Enterprise-Grade Operational Mode System --- #
class OperationalMode(str, Enum):
    """Standardized operational modes for the trading system."""

    LIVE_TRADING = "live_trading"
    PAPER_TRADING = "paper_trading"
    BACKTESTING = "backtesting"
    DATA_COLLECTION = "data_collection"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MONITORING_ONLY = "monitoring_only"
    MAINTENANCE = "maintenance"
    RESEARCH = "research"
    DEVELOPMENT = "development"


class EnvironmentType(str, Enum):
    """Environment types for deployment contexts."""

    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    LOCAL = "local"


@dataclass
class ModeCapabilities:
    """Capabilities and requirements for each operational mode."""

    requires_market_data: bool
    requires_execution_handler: bool
    requires_portfolio_manager: bool
    requires_risk_manager: bool
    requires_prediction_service: bool
    supports_real_money: bool
    resource_intensity: str  # 'low', 'medium', 'high'
    description: str


@dataclass
class ModeValidationResult:
    """Result of operational mode validation."""

    is_valid: bool
    mode: OperationalMode
    environment: EnvironmentType
    validation_checks: dict[str, bool]
    warnings: list[str]
    errors: list[str]
    missing_requirements: list[str]


class ModeConfiguration(BaseModel):
    """Configuration for operational modes."""

    supported_modes: list[OperationalMode] = Field(
        description="List of supported operational modes",
    )

    default_mode: OperationalMode = Field(
        default=OperationalMode.PAPER_TRADING,
        description="Default mode if none specified",
    )

    mode_capabilities: dict[str, dict[str, Any]] = Field(
        default_factory=dict[str, Any],
        description="Capabilities required for each mode",
    )

    incompatible_combinations: list[list[OperationalMode]] = Field(
        default_factory=list[Any],
        description="Mode combinations that are not allowed",
    )

    environment_restrictions: dict[str, list[OperationalMode]] = Field(
        default_factory=dict[str, Any],
        description="Mode restrictions per environment (dev, staging, prod)",
    )

    @field_validator("supported_modes")
    @classmethod
    def validate_supported_modes(cls, v: list[str]) -> list[str]:
        """Validate that supported modes are valid and non-empty."""
        if not v:
            raise ValueError("At least one operational mode must be supported")

        # Ensure all modes are valid enum values
        valid_modes = set(OperationalMode)
        for mode in v:
            if mode not in valid_modes:
                raise ValueError(f"Invalid mode: {mode}")

        return v

    @field_validator("default_mode")
    @classmethod
    def validate_default_mode(cls, v: str, info: Any) -> str:
        """Validate that default mode is in supported modes."""
        if info.data and "supported_modes" in info.data:
            supported_modes = info.data["supported_modes"]
            if v not in supported_modes:
                raise ValueError("Default mode must be in supported modes list[Any]")
        return v


class OperationalModeManager:
    """Enterprise-grade operational mode management system."""

    def __init__(self, config_manager: ConfigManagerType) -> None:
        """Initialize the mode manager with configuration."""
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.mode_config = self._load_mode_configuration()
        self.current_mode: OperationalMode | None = None
        self.current_environment: EnvironmentType | None = None

        # Define default capabilities for each mode
        self.default_capabilities = {
            OperationalMode.LIVE_TRADING: ModeCapabilities(
                requires_market_data=True,
                requires_execution_handler=True,
                requires_portfolio_manager=True,
                requires_risk_manager=True,
                requires_prediction_service=True,
                supports_real_money=True,
                resource_intensity="high",
                description="Full live trading with real money",
            ),
            OperationalMode.PAPER_TRADING: ModeCapabilities(
                requires_market_data=True,
                requires_execution_handler=True,
                requires_portfolio_manager=True,
                requires_risk_manager=True,
                requires_prediction_service=True,
                supports_real_money=False,
                resource_intensity="medium",
                description="Simulated trading with fake money",
            ),
            OperationalMode.BACKTESTING: ModeCapabilities(
                requires_market_data=True,
                requires_execution_handler=False,
                requires_portfolio_manager=True,
                requires_risk_manager=True,
                requires_prediction_service=True,
                supports_real_money=False,
                resource_intensity="high",
                description="Historical strategy testing",
            ),
            OperationalMode.DATA_COLLECTION: ModeCapabilities(
                requires_market_data=True,
                requires_execution_handler=False,
                requires_portfolio_manager=False,
                requires_risk_manager=False,
                requires_prediction_service=False,
                supports_real_money=False,
                resource_intensity="low",
                description="Market data collection only",
            ),
            OperationalMode.MONITORING_ONLY: ModeCapabilities(
                requires_market_data=True,
                requires_execution_handler=False,
                requires_portfolio_manager=True,
                requires_risk_manager=True,
                requires_prediction_service=False,
                supports_real_money=False,
                resource_intensity="low",
                description="Monitor positions without trading",
            ),
            OperationalMode.MODEL_TRAINING: ModeCapabilities(
                requires_market_data=True,
                requires_execution_handler=False,
                requires_portfolio_manager=False,
                requires_risk_manager=False,
                requires_prediction_service=True,
                supports_real_money=False,
                resource_intensity="high",
                description="Train predictive models",
            ),
            OperationalMode.RESEARCH: ModeCapabilities(
                requires_market_data=True,
                requires_execution_handler=False,
                requires_portfolio_manager=False,
                requires_risk_manager=False,
                requires_prediction_service=False,
                supports_real_money=False,
                resource_intensity="medium",
                description="Research and analysis mode",
            ),
        }

    def get_supported_modes(self) -> list[OperationalMode]:
        """Get supported modes from configuration, replacing hardcoded examples.

        This replaces line 709: supported_modes=["live", "paper"] with configuration-driven values.
        """
        try:
            # Load from configuration
            supported_modes = self.mode_config.supported_modes

            # Apply environment restrictions
            current_env = self.config_manager.get("environment", "development")
            if current_env in self.mode_config.environment_restrictions:
                allowed_modes = self.mode_config.environment_restrictions[current_env]
                supported_modes = [mode for mode in supported_modes if mode in allowed_modes]

            self.logger.info(
                f"Loaded {len(supported_modes)} supported modes from configuration: "
                f"{[mode.value for mode in supported_modes]}",
            )

        except Exception:
            self.logger.exception("Error loading supported modes from configuration: ")
            # Fallback to safe default
            fallback_modes = [OperationalMode.PAPER_TRADING, OperationalMode.DATA_COLLECTION]
            self.logger.warning(f"Using fallback modes: {[mode.value for mode in fallback_modes]}")
            return fallback_modes
        else:
            return supported_modes

    def detect_environment(self) -> EnvironmentType:
        """Detect current environment based on various indicators."""
        # Check environment variable first
        env_var = os.getenv("GAL_FRIDAY_ENV", "").lower()
        if env_var:
            try:
                return EnvironmentType(env_var)
            except ValueError:
                self.logger.warning(f"Invalid environment variable value: {env_var}")

        # Check configuration file
        config_env = self.config_manager.get("environment", "").lower()
        if config_env:
            try:
                return EnvironmentType(config_env)
            except ValueError:
                self.logger.warning(f"Invalid environment in config: {config_env}")

        # Auto-detect based on system characteristics
        if self._is_production_environment():
            return EnvironmentType.PRODUCTION
        if self._is_staging_environment():
            return EnvironmentType.STAGING
        if self._is_development_environment():
            return EnvironmentType.DEVELOPMENT
        return EnvironmentType.LOCAL

    async def select_and_initialize_operational_mode(
        self,
        explicit_mode: str | None = None,
    ) -> OperationalMode:
        """Select and initialize the appropriate operational mode."""
        try:
            self.logger.info("Starting operational mode selection and initialization")

            # Detect current environment
            environment = self.detect_environment()
            self.logger.info(f"Detected environment: {environment.value}")

            # Determine operational mode
            mode = await self._determine_operational_mode(environment, explicit_mode)
            self.logger.info(f"Selected operational mode: {mode.value}")

            # Validate mode compatibility
            validation_result = await self._validate_mode_compatibility(mode, environment)

            if not validation_result.is_valid:
                error_msg = f"Mode validation failed: {validation_result.errors}"
                self.logger.error(error_msg)
                self._raise_value_error(error_msg)

            # Log warnings if any
            for warning in validation_result.warnings:
                self.logger.warning(f"Mode validation warning: {warning}")

            # Set current mode and environment
            self.current_mode = mode
            self.current_environment = environment

            self.logger.info(
                f"Operational mode initialization complete: {mode.value} in {environment.value} environment",
            )

        except Exception:
            self.logger.exception("Error in operational mode selection: ")
            raise
        else:
            return mode

    async def _determine_operational_mode(
        self,
        environment: EnvironmentType,
        explicit_mode: str | None = None,
    ) -> OperationalMode:
        """Determine operational mode based on environment and configuration."""
        # Check explicit mode configuration first
        if explicit_mode:
            try:
                mode = OperationalMode(explicit_mode)
                self.logger.debug(f"Using explicit mode configuration: {mode.value}")
            except ValueError:
                self.logger.warning(f"Invalid explicit mode configuration: {explicit_mode}")
            else:
                return mode

        # Check configuration file mode
        config_mode = self.config_manager.get("operational_mode", "").lower()
        if config_mode:
            try:
                mode = OperationalMode(config_mode)
                self.logger.debug(f"Using configuration file mode: {mode.value}")
            except ValueError:
                self.logger.warning(f"Invalid mode in configuration: {config_mode}")
            else:
                return mode

        # Environment-based defaults
        if environment == EnvironmentType.PRODUCTION:
            return OperationalMode.LIVE_TRADING
        if environment in (EnvironmentType.STAGING, EnvironmentType.DEVELOPMENT):
            return OperationalMode.PAPER_TRADING
        # LOCAL
        return OperationalMode.BACKTESTING

    async def _validate_mode_compatibility(
        self,
        mode: OperationalMode,
        environment: EnvironmentType,
    ) -> ModeValidationResult:
        """Validate that the selected mode is compatible with current environment."""
        validation_checks: dict[str, Any] = {}
        warnings: list[str] = []
        errors: list[str] = []
        missing_requirements: list[str] = []

        # Get mode capabilities
        capabilities = self.default_capabilities.get(mode)
        if not capabilities:
            errors.append(f"No capabilities defined for mode: {mode.value}")
            return ModeValidationResult(
                is_valid=False,
                mode=mode,
                environment=environment,
                validation_checks=validation_checks,
                warnings=warnings,
                errors=errors,
                missing_requirements=missing_requirements,
            )

        # Validate environment-specific restrictions
        if environment == EnvironmentType.PRODUCTION and not capabilities.supports_real_money:
            warnings.append("Non-money mode in production environment")

        # Basic capability checks (simplified - would need actual service availability checks)
        validation_checks["market_data"] = capabilities.requires_market_data
        validation_checks["execution_handler"] = capabilities.requires_execution_handler
        validation_checks["portfolio_manager"] = capabilities.requires_portfolio_manager
        validation_checks["risk_manager"] = capabilities.requires_risk_manager
        validation_checks["prediction_service"] = capabilities.requires_prediction_service

        is_valid = len(errors) == 0

        return ModeValidationResult(
            is_valid=is_valid,
            mode=mode,
            environment=environment,
            validation_checks=validation_checks,
            warnings=warnings,
            errors=errors,
            missing_requirements=missing_requirements,
        )

    def validate_mode_request(self, requested_modes: list[str]) -> list[OperationalMode]:
        """Validate requested modes against configuration and compatibility rules."""
        if not requested_modes:
            default_mode = self.mode_config.default_mode
            self.logger.info(f"No modes specified, using default: {default_mode.value}")
            return [default_mode]

        # Convert strings to enum values
        validated_modes = []
        supported_modes = self.get_supported_modes()

        for mode_str in requested_modes:
            try:
                mode = OperationalMode(mode_str)

                # Check if mode is supported
                if mode not in supported_modes:
                    self._raise_value_error(
                        f"Mode '{mode_str}' is not supported in current configuration",
                    )

                validated_modes.append(mode)

            except ValueError as e:
                available_modes = [mode.value for mode in supported_modes]
                raise ValueError(
                    f"Invalid mode '{mode_str}'. Available modes: {available_modes}",
                ) from e

        # Check for incompatible combinations
        self._validate_mode_compatibility_rules(validated_modes)

        return validated_modes

    def _validate_mode_compatibility_rules(self, modes: list[OperationalMode]) -> None:
        """Check for incompatible mode combinations."""
        mode_set = set(modes)

        # Check against configured incompatible combinations
        for incompatible_group in self.mode_config.incompatible_combinations:
            if len(mode_set.intersection(set(incompatible_group))) > 1:
                conflicting = mode_set.intersection(set(incompatible_group))
                raise ValueError(
                    f"Incompatible mode combination detected: {[m.value for m in conflicting]}",
                )

        # Built-in compatibility checks
        if OperationalMode.LIVE_TRADING in mode_set and OperationalMode.BACKTESTING in mode_set:
            raise ValueError("Cannot run live trading and backtesting simultaneously")

        if OperationalMode.LIVE_TRADING in mode_set and OperationalMode.PAPER_TRADING in mode_set:
            raise ValueError("Cannot run live trading and paper trading simultaneously")

    def get_required_services(self, modes: list[OperationalMode]) -> dict[str, bool]:
        """Determine which services are required for the given modes."""
        requirements = {
            "market_data": False,
            "execution_handler": False,
            "portfolio_manager": False,
            "risk_manager": False,
            "prediction_service": False,
        }

        for mode in modes:
            capabilities = self.default_capabilities.get(mode)
            if capabilities:
                requirements["market_data"] |= capabilities.requires_market_data
                requirements["execution_handler"] |= capabilities.requires_execution_handler
                requirements["portfolio_manager"] |= capabilities.requires_portfolio_manager
                requirements["risk_manager"] |= capabilities.requires_risk_manager
                requirements["prediction_service"] |= capabilities.requires_prediction_service

        return requirements

    def _load_mode_configuration(self) -> ModeConfiguration:
        """Load mode configuration from ConfigManager."""
        try:
            mode_config_dict = self.config_manager.get("operational_modes", {})

            # Provide defaults if configuration is incomplete
            if not mode_config_dict.get("supported_modes"):
                mode_config_dict["supported_modes"] = [
                    OperationalMode.PAPER_TRADING,
                    OperationalMode.BACKTESTING,
                    OperationalMode.DATA_COLLECTION,
                ]

            return ModeConfiguration(**mode_config_dict)

        except Exception:
            self.logger.exception("Error loading mode configuration: ")
            # Return safe default configuration
            return ModeConfiguration(
                supported_modes=[OperationalMode.PAPER_TRADING, OperationalMode.DATA_COLLECTION],
                default_mode=OperationalMode.PAPER_TRADING,
            )

    def _is_production_environment(self) -> bool:
        """Check if running in production environment."""
        production_indicators = [
            os.getenv("IS_PRODUCTION", "").lower() == "true",
            Path("/etc/gal-friday/production.conf").exists(),
            "production" in str(Path.cwd()).lower(),
            os.getenv("DEPLOYMENT_ENV", "").lower() == "production",
        ]
        return any(production_indicators)

    def _is_staging_environment(self) -> bool:
        """Check if running in staging environment."""
        staging_indicators = [
            os.getenv("IS_STAGING", "").lower() == "true",
            Path("/etc/gal-friday/staging.conf").exists(),
            "staging" in str(Path.cwd()).lower(),
            os.getenv("DEPLOYMENT_ENV", "").lower() == "staging",
        ]
        return any(staging_indicators)

    def _is_development_environment(self) -> bool:
        """Check if running in development environment."""
        development_indicators = [
            os.getenv("IS_DEVELOPMENT", "").lower() == "true",
            Path("pyproject.toml.old").exists(),  # Development workspace indicator
            ".git" in [p.name for p in Path().iterdir()] if Path().exists() else False,
            os.getenv("DEPLOYMENT_ENV", "").lower() == "development",
        ]
        return any(development_indicators)

    def _raise_value_error(self, error_msg: str) -> None:
        """Helper method to raise ValueError, abstracting the raise from try block."""
        raise ValueError(error_msg)


# --- Enhanced Logging Setup System --- #
class LoggingSetup:
    """Enterprise-grade logging configuration and initialization."""

    def __init__(self, config: ConfigManagerType) -> None:
        """Initialize logging setup with configuration."""
        self.config = config
        self.logger: logging.Logger | None = None

    def setup_logging(
        self,
        log_level: str | None = None,
        log_file: str | None = None,
        enable_json: bool = False,
        enable_console: bool = True,
    ) -> logging.Logger:
        """Setup comprehensive logging system with multiple handlers and formatters."""
        # Determine log level
        if log_level:
            level = getattr(logging, log_level.upper(), logging.INFO)
        else:
            level = getattr(
                logging,
                self.config.get("logging", {}).get("level", "INFO").upper(),
                logging.INFO,
            )

        # Create formatters
        formatters = self._create_formatters(enable_json)

        # Create handlers
        handlers = self._create_handlers(log_file, enable_console, formatters)

        # Configure root logger
        logging.basicConfig(
            level=level,
            handlers=handlers,
            force=True,  # Override any existing configuration
        )

        # Configure specific loggers
        self._configure_library_loggers()

        # Create application logger
        logger = logging.getLogger("gal_friday")
        logger.setLevel(level)

        # Log initial startup information
        logger.info(f"Logging initialized - Level: {logging.getLevelName(level)}")
        logger.info(f"Log handlers: {[type(h).__name__ for h in handlers]}")

        self.logger = logger
        return logger

    def _create_formatters(self, enable_json: bool) -> dict[str, logging.Formatter]:
        """Create logging formatters."""
        formatters = {}

        # Standard formatter
        formatters["standard"] = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Detailed formatter
        formatters["detailed"] = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # JSON formatter (simplified implementation)
        if enable_json:
            formatters["json"] = logging.Formatter(
                '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        return formatters

    def _create_handlers(
        self,
        log_file: str | None,
        enable_console: bool,
        formatters: dict[str, logging.Formatter],
    ) -> list[logging.Handler]:
        """Create logging handlers."""
        handlers: list[logging.Handler] = []

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatters["standard"])
            handlers.append(console_handler)

        # File handler with rotation
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
            )
            file_handler.setFormatter(formatters["detailed"])
            handlers.append(file_handler)

        # Error file handler
        if log_file:
            error_log_file = str(log_path.parent / f"{log_path.stem}_errors{log_path.suffix}")
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=3,
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatters["detailed"])
            handlers.append(error_handler)

        return handlers

    def _configure_library_loggers(self) -> None:
        """Configure third-party library loggers to reduce noise."""
        # Reduce verbosity of common libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.INFO)


# --- Enhanced CLI Parser System --- #
class EnhancedCLIParser:
    """Command line interface parser with comprehensive options."""

    def __init__(self) -> None:
        """Initialize the CLI parser."""
        self.parser: argparse.ArgumentParser | None = None
        self._setup_parser()

    def _setup_parser(self) -> None:
        """Setup command line argument parser with comprehensive options."""
        self.parser = argparse.ArgumentParser(
            description="Gal Friday - Enterprise Trading System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s --mode live_trading --config config/production.yaml
  %(prog)s --mode paper_trading --log-level DEBUG
  %(prog)s --mode backtesting --log-file logs/backtest.log
  %(prog)s --validate-config --config config/staging.yaml
            """,
        )

        # Operational mode arguments
        mode_group = self.parser.add_argument_group("Operational Mode")
        mode_group.add_argument(
            "--mode",
            "-m",
            choices=[mode.value for mode in OperationalMode],
            default=OperationalMode.PAPER_TRADING.value,
            help="Operational mode (default: paper_trading)",
        )

        # Configuration arguments
        config_group = self.parser.add_argument_group("Configuration")
        config_group.add_argument(
            "--config",
            "-c",
            type=str,
            default="config/default.yaml",
            help="Path to configuration file (default: config/default.yaml)",
        )
        config_group.add_argument(
            "--validate-config",
            action="store_true",
            help="Validate configuration and exit",
        )

        # Logging arguments
        log_group = self.parser.add_argument_group("Logging")
        log_group.add_argument(
            "--log-level",
            "-l",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set logging level",
        )
        log_group.add_argument("--log-file", type=str, help="Path to log file")
        log_group.add_argument(
            "--enable-json-logs",
            action="store_true",
            help="Enable JSON structured logging",
        )
        log_group.add_argument(
            "--disable-console-logs",
            action="store_true",
            help="Disable console logging output",
        )

        # Environment and deployment
        env_group = self.parser.add_argument_group("Environment")
        env_group.add_argument(
            "--environment",
            choices=[env.value for env in EnvironmentType],
            help="Deployment environment",
        )

        # Application control
        control_group = self.parser.add_argument_group("Application Control")
        control_group.add_argument(
            "--dry-run",
            action="store_true",
            help="Perform dry run without executing trades",
        )
        control_group.add_argument(
            "--health-check",
            action="store_true",
            help="Perform health check and exit",
        )
        control_group.add_argument(
            "--version",
            "-V",
            action="version",
            version=f"Gal Friday Trading System v{__version__}",
        )

    def parse_args(self, args: list[str] | None = None) -> argparse.Namespace:
        """Parse command line arguments with validation."""
        if self.parser is None:
            raise RuntimeError("Parser not initialized")
        parsed_args = self.parser.parse_args(args)

        # Validate arguments
        self._validate_args(parsed_args)

        return parsed_args

    def _validate_args(self, parsed_args: argparse.Namespace) -> None:
        """Validate parsed arguments."""
        # Validate config file exists if specified
        if (
            parsed_args.config and not Path(parsed_args.config).exists() and not parsed_args.validate_config
        ):  # Don't fail if just validating
            if self.parser is None:
                raise RuntimeError("Parser not initialized")
            self.parser.error(f"Configuration file not found: {parsed_args.config}")

        # Validate log file directory exists or can be created
        if parsed_args.log_file:
            log_dir = Path(parsed_args.log_file).parent
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                if self.parser is None:
                    raise RuntimeError("Parser not initialized") from e
                self.parser.error(f"Cannot create log directory {log_dir}: {e}")


def initialize_application_components(
    config_manager: ConfigManagerType,
) -> tuple[logging.Logger, argparse.Namespace, OperationalModeManager]:
    """Initialize application with enhanced logging, CLI setup, and mode management."""
    # Setup command line interface
    cli_parser = EnhancedCLIParser()
    args = cli_parser.parse_args()

    # Setup logging
    logging_setup = LoggingSetup(config_manager)
    logger = logging_setup.setup_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        enable_json=args.enable_json_logs,
        enable_console=not args.disable_console_logs,
    )

    # Create operational mode manager
    mode_manager = OperationalModeManager(config_manager)

    # Log startup information
    logger.info("Gal Friday Trading System Starting")
    logger.info(f"Version: {__version__}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Environment: {getattr(args, 'environment', 'auto-detect')}")

    # Log available modes for transparency
    supported_modes = mode_manager.get_supported_modes()
    logger.info(f"Application supports {len(supported_modes)} operational modes:")
    for mode in supported_modes:
        capabilities = mode_manager.default_capabilities.get(mode)
        if capabilities:
            logger.info(f"  - {mode.value}: {capabilities.description}")

    return logger, args, mode_manager


# --- Conditional Imports for Type[Any] Checking --- #
if TYPE_CHECKING:
    # ERA001: Removed commented-out import.
    # Define a proper protocol/interface for execution handlers
    from typing import Protocol

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker  # Added

    from .core.initialization_patterns import (
        TypeSafeInitializationManager,
    )
    from .core.pubsub import PubSubManager as PubSubManagerType
    from .data_ingestor import DataIngestor as DataIngestorType

    # Ensure these are imported if they are to be used as string literals in hints
    from .execution.kraken import KrakenExecutionHandler
    from .feature_engine import FeatureEngine as FeatureEngineType
    from .historical_data_service import (
        HistoricalDataService as HistoricalDataServiceType,
    )
    from .interfaces.service_protocol import ServiceProtocol
    from .logger_service import LoggerService as LoggerServiceType
    from .market_price_service import MarketPriceService as MarketPriceServiceType
    from .monitoring_service import MonitoringService as MonitoringServiceType
    from .portfolio_manager import PortfolioManager as PortfolioManagerType
    from .prediction_service import PredictionService as PredictionServiceType
    from .risk_manager import RiskManager as RiskManagerType
    from .simulated_execution_handler import SimulatedExecutionHandler

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

            Args:
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

            Args:
            ----
                order_data: Dictionary containing order details

            Returns:
            -------
                Order ID from the exchange
            """
            ...

        def cancel_order(self, order_id: str) -> bool:
            """Cancel an existing order.

            Args:
            ----
                order_id: ID of the order to cancel

            Returns:
            -------
                True if cancellation was successful, False otherwise
            """
            ...

    # Now use this protocol for type annotations
    ExecutionHandlerTypeHint = type[ExecutionHandlerProtocol]
    _ExecutionHandlerType = KrakenExecutionHandler | SimulatedExecutionHandler


# --- Attempt to import core application modules (Runtime) --- #
# Initialize basic logging for import issues
startup_logger = logging.getLogger("gal_friday.startup")

try:
    from .config_manager import ConfigManager
except ImportError:
    startup_logger.exception("Failed to import ConfigManager")
    ConfigManager = None  # type: ignore[assignment,misc]

try:
    from .core.pubsub import PubSubManager
except ImportError:
    startup_logger.exception("Failed to import PubSubManager")
    PubSubManager = None  # type: ignore[assignment,misc]

try:
    from .data_ingestor import DataIngestor
except ImportError:
    startup_logger.exception("Failed to import DataIngestor")
    DataIngestor = None  # type: ignore[assignment,misc]

try:
    from .prediction_service import PredictionService
except ImportError:
    startup_logger.exception("Failed to import PredictionService")
    PredictionService = None  # type: ignore[assignment,misc]

try:
    from .strategy_arbitrator import StrategyArbitrator
except ImportError:
    startup_logger.exception("Failed to import StrategyArbitrator")
    StrategyArbitrator = None  # type: ignore[assignment,misc]

try:
    from .portfolio_manager import PortfolioManager
except ImportError:
    startup_logger.exception("Failed to import PortfolioManager")
    PortfolioManager = None  # type: ignore[assignment,misc]

try:
    from .risk_manager import RiskManager
except ImportError:
    startup_logger.exception("Failed to import RiskManager")
    RiskManager = None  # type: ignore[assignment,misc]

# --- Execution Handler Imports (Runtime) --- #
try:
    from .execution.kraken import KrakenExecutionHandler
except ImportError:
    startup_logger.exception("Failed to import KrakenExecutionHandler: ")
    KrakenExecutionHandler = None  # type: ignore[assignment,misc]

try:
    from .simulated_execution_handler import SimulatedExecutionHandler
except ImportError:
    startup_logger.exception("Failed to import SimulatedExecutionHandler")
    SimulatedExecutionHandler = None  # type: ignore[assignment,misc]

# --- Other Service Imports (Runtime) --- #
try:
    from .logger_service import LoggerService
except ImportError:
    startup_logger.exception("Failed to import LoggerService")
    LoggerService = None  # type: ignore[assignment,misc]

try:
    from .monitoring_service import MonitoringService
except ImportError:
    startup_logger.exception("Failed to import MonitoringService")
    MonitoringService = None  # type: ignore[assignment,misc]

try:
    from .cli_service import CLIService
except ImportError:
    startup_logger.exception("Failed to import CLIService")
    CLIService = None  # type: ignore[assignment,misc]

try:
    from .market_price_service import MarketPriceService
except ImportError:
    startup_logger.exception("Failed to import MarketPriceService")
    MarketPriceService = None  # type: ignore[assignment,misc]

try:
    from .historical_data_service import HistoricalDataService
except ImportError:
    startup_logger.exception("Failed to import HistoricalDataService")
    HistoricalDataService = None  # type: ignore[assignment,misc]

try:
    from .core.feature_registry_client import FeatureRegistryClient
except ImportError:
    startup_logger.exception("Failed to import FeatureRegistryClient")
    FeatureRegistryClient = None  # type: ignore[assignment,misc]

# --- DAL Imports ---
try:
    from .dal.connection_pool import DatabaseConnectionPool
except ImportError:
    startup_logger.exception("Failed to import DatabaseConnectionPool")
    DatabaseConnectionPool = None  # type: ignore[assignment,misc]

try:
    from .dal.migrations.migration_manager import MigrationManager
except ImportError:
    startup_logger.exception("Failed to import MigrationManager")
    MigrationManager = None  # type: ignore[assignment,misc]


# --- Import concrete service implementations --- #
try:
    from .market_price.kraken_service import KrakenMarketPriceService
except ImportError as e:
    startup_logger.warning("Failed to import KrakenMarketPriceService: %s", e)
    KrakenMarketPriceService = None  # type: ignore[assignment,misc]

try:
    from .kraken_historical_data_service import KrakenHistoricalDataService
except ImportError as e:
    startup_logger.warning("Failed to import KrakenHistoricalDataService: %s", e)
    KrakenHistoricalDataService = None  # type: ignore[assignment,misc]

try:
    from .simulated_market_price_service import (  # Restored runtime import
        SimulatedMarketPriceService,
    )
except ImportError:
    startup_logger.exception("Failed to import SimulatedMarketPriceService")
    SimulatedMarketPriceService = None  # type: ignore[assignment,misc] # Restored fallback

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
def setup_logging(config: Optional["ConfigManagerType"], log_level_override: str | None = None) -> None:
    """Configure logging based on the application configuration."""
    # Runtime check still needed
    if config is None or ConfigManager is None:
        log.warning("ConfigManager instance or class not available, cannot configure logging.")
        return

    # No assertion needed here as we checked config is not None
    log_config = config.get("logging", {})
    log_level_name = log_level_override or log_config.get("level", "INFO").upper()  # Use override if provided
    log_level = getattr(logging, log_level_name, logging.INFO)

    root_logger = _setup_root_logger(log_level, log_level_name)

    # Set up handlers
    _setup_console_handler(root_logger, log_config, log_level)
    _setup_json_file_handler(root_logger, log_config, log_level)
    _setup_database_handler(log_config)


def _setup_root_logger(log_level: int, log_level_name: str) -> logging.Logger:
    """Set up the root logger with the specified level."""
    root_logger = logging.getLogger()  # Get the root logger
    root_logger.setLevel(log_level)

    # Clear existing handlers (e.g., from basicConfig)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    log.info("Root logger level set to %s", log_level_name)
    return root_logger


def _setup_console_handler(root_logger: logging.Logger, log_config: dict[str, Any], log_level: int) -> None:
    """Set up console logging handler if enabled."""
    console_config = log_config.get("console", {})
    if not console_config.get("enabled", True):
        return

    console_format = console_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    # Handler level defaults to root logger level
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(console_format, datefmt=log_config.get("date_format"))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    log.info("Console logging enabled.")


def _setup_json_file_handler(root_logger: logging.Logger, log_config: dict[str, Any], log_level: int) -> None:
    """Set up JSON file logging handler if enabled."""
    json_file_config = log_config.get("json_file", {})
    if not json_file_config.get("enabled", False):
        return

    log_filename = json_file_config.get("filename")
    if not log_filename:
        log.warning("File logging enabled in config but no filename specified.")
        return

    # Ensure log directory exists
    log_filename = _ensure_log_directory(log_filename)
    if not log_filename:
        return

    # Create and configure the file handler
    file_handler = _create_file_handler(json_file_config, log_filename)
    file_handler.setLevel(log_level)

    # Set up formatter
    formatter = _create_file_formatter(json_file_config, log_config)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    use_json = json_file_config.get("use_json", True)
    log.info("File logging enabled: %s (JSON=%s)", log_filename, use_json)


def _ensure_log_directory(log_filename: str) -> str | None:
    """Ensure the log directory exists, create if necessary."""
    log_dir_path = Path(log_filename).parent
    if log_dir_path and not log_dir_path.exists():
        try:
            log_dir_path.mkdir(parents=True, exist_ok=True)
            log.info("Created log directory: %s", log_dir_path)
        except OSError:
            log.exception("Could not create log directory")
            return None  # Prevent handler creation if dir fails
    return log_filename


def _create_file_handler(json_file_config: dict[str, Any], log_filename: str) -> logging.handlers.RotatingFileHandler:
    """Create a rotating file handler with configured settings."""
    max_bytes = json_file_config.get("max_bytes", 10 * 1024 * 1024)  # Default 10MB
    backup_count = json_file_config.get("backup_count", 5)

    return logging.handlers.RotatingFileHandler(
        log_filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )


def _create_file_formatter(json_file_config: dict[str, Any], log_config: dict[str, Any]) -> logging.Formatter:
    """Create appropriate formatter based on configuration."""
    file_format = json_file_config.get("format", "%(asctime)s %(name)s %(levelname)s %(message)s")
    use_json = json_file_config.get("use_json", True)

    if use_json:
        return jsonlogger.JsonFormatter(file_format)
    return logging.Formatter(file_format, datefmt=log_config.get("date_format"))


def _setup_database_handler(log_config: dict[str, Any]) -> None:
    """Set up database logging handler if enabled."""
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

        # Enterprise dependency injection container
        self.container: ServiceContainer | None = None

        # Type-safe initialization manager
        self.initialization_manager: TypeSafeInitializationManager | None = None

        # Service initialization director
        self.service_director: ServiceInitializationDirector | None = None

        # Core application state
        self.config: ConfigManagerType | None = None
        self.pubsub: PubSubManagerType | None = None
        self.executor: concurrent.futures.ProcessPoolExecutor | None = None
        self.services: list[ServiceProtocol] = []
        self.running_tasks: list[asyncio.Task[Any]] = []  # UP006: List -> list[Any]
        self.args: argparse.Namespace | None = None

        # Enterprise-grade components
        self.mode_manager: OperationalModeManager | None = None
        self.logging_setup: LoggingSetup | None = None
        self.current_operational_mode: OperationalMode | None = None

        # Store references to specific services after instantiation for DI
        self.logger_service: LoggerServiceType | None = None
        self.db_connection_pool: DatabaseConnectionPool | None = None  # Added
        self.session_maker: async_sessionmaker[AsyncSession] | None = None
        self.migration_manager: MigrationManager | None = None  # Added
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
        self.feature_registry_client: FeatureRegistryClient | None = None  # Added

        # Keep a direct reference to config_manager if needed by other methods
        self._config_manager_instance: ConfigManagerType | None = None

    def _load_configuration(self, config_path: str) -> None:
        """Load the application configuration using type-safe initialization."""
        try:
            self._ensure_class_available(ConfigManager, "ConfigManager", "Configuration loading")

            # Use type-safe configuration initialization
            self._config_manager_instance = ConfigManager(
                config_path=config_path,
                logger_service=logging.getLogger("gal_friday.config_manager"),
            )

            # Validate configuration with type safety
            self.config = ensure_config_manager(self._config_manager_instance)

            log.info("Configuration loaded successfully from: %s", config_path)
            if not self.config.is_valid():
                log.error("Initial configuration is invalid. Review errors logged by ConfigManager.")
                # Depending on desired strictness, could raise ConfigurationLoadingFailedExit here.
        except Exception as e:
            log.exception("FATAL: Failed to load configuration from %s", config_path)
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
        error_msg = "ConfigManager instance not available for PubSubManager. Load configuration first."
        log.critical(error_msg)
        raise DependencyMissingError("PubSubManager", error_msg)

    def _update_config_manager_with_pubsub(self) -> None:
        """Update ConfigManager with PubSubManager instance if available."""
        if self._config_manager_instance and hasattr(self._config_manager_instance, "set_pubsub_manager"):
            self._config_manager_instance.set_pubsub_manager(self.pubsub)
        if hasattr(self, "_logger"):
            self._logger.info("Updated ConfigManager with PubSubManager instance.")

    def _instantiate_pubsub(self) -> None:
        """Instantiate the PubSubManager using type-safe patterns."""
        try:
            self._ensure_class_available(PubSubManager, "PubSubManager", "PubSubManager instantiation")

            # Enhanced type-safe dependency validation
            config = ensure_config_manager(self.config)

            # Validate configuration has required methods
            TypeSafeAssertion.assert_has_methods(config, ["get"], "ConfigManager for PubSub")

            self.pubsub = PubSubManager(logger=logging.getLogger("gal_friday.pubsub"), config_manager=config)

            # Validate the created PubSubManager
            self.pubsub = TypeSafeAssertion.assert_pubsub_manager(
                self.pubsub,
                "PubSubManager instantiation",
            )

            log.info("PubSubManager instantiated successfully with type safety.")

            # Now that pubsub is available, update config_manager if needed
            self._update_config_manager_with_pubsub()

        except Exception as e:
            log.exception("FATAL: Failed to instantiate PubSubManager")
            raise PubSubManagerInstantiationFailedExit from e

    def _init_strategy_arbitrator(self) -> StrategyArbitratorType:
        """Instantiate the StrategyArbitrator."""
        # Prerequisites will be checked in this order and fail fast
        self._ensure_strategy_arbitrator_prerequisites()

        # Type-safe dependency validation
        config = DependencyValidator.assert_config_manager(self.config)
        pubsub = DependencyValidator.assert_pubsub_manager(self.pubsub)
        logger = DependencyValidator.assert_logger_service(self.logger_service)

        strategy_arbitrator_config = config.get("strategy_arbitrator", {})
        if self.market_price_service is None:
            raise MarketPriceServiceError

        # Create FeatureRegistryClient instance
        if FeatureRegistryClient is not None:
            self.feature_registry_client = FeatureRegistryClient()
            log.debug("FeatureRegistryClient instantiated.")
        else:
            # mypy doesn't understand runtime import patterns
            log.warning(  # type: ignore[unreachable]
                "FeatureRegistryClient not available, StrategyArbitrator will operate without feature validation.",
            )
            self.feature_registry_client = None

        self.strategy_arbitrator = StrategyArbitrator(
            config=strategy_arbitrator_config,
            pubsub_manager=pubsub,
            logger_service=logger,
            market_price_service=self.market_price_service,
            feature_registry_client=self.feature_registry_client,
        )
        log.debug("StrategyArbitrator instantiated.")
        return self.strategy_arbitrator

    def _init_cli_service(self) -> CLIServiceType | None:
        """Instantiate the CLIService."""
        if CLIService is not None:
            # Prerequisite checks for CLIService (monitoring_service, portfolio_manager)
            if self.monitoring_service is None:
                raise DependencyMissingError(component="CLIService", dependency="MonitoringService instance")
            if self.portfolio_manager is None:
                raise DependencyMissingError(component="CLIService", dependency="PortfolioManager instance")

            # Type-safe dependency validation
            logger = DependencyValidator.assert_logger_service(self.logger_service)

            self.cli_service = CLIService(
                monitoring_service=self.monitoring_service,
                logger_service=logger,
                main_app_controller=self,
                portfolio_manager=self.portfolio_manager,
            )
            log.debug("CLIService instantiated.")
            return self.cli_service
        # mypy doesn't understand runtime import patterns
        self.cli_service = None  # type: ignore[unreachable]
        log.info("CLIService not available or not configured.")
        return None

    def _ensure_class_available(
        self,
        class_obj: type | None,
        class_name_str: str,
        required_by_component: str = "GalFridayApp",
    ) -> None:
        """Ensure a class is available (not None), raises DependencyMissingError if not."""
        if class_obj is None:
            raise DependencyMissingError(component=required_by_component, dependency="Required class")

    def _ensure_strategy_arbitrator_prerequisites(self) -> None:
        """Ensure all prerequisites for StrategyArbitrator are met before instantiation."""
        if self.market_price_service is None:
            raise DependencyMissingError(component="StrategyArbitrator", dependency="MarketPriceService instance")

    def _ensure_risk_manager_prerequisites(self) -> None:
        """Ensure all prerequisites for RiskManager are met before instantiation."""
        if self.market_price_service is None:
            raise DependencyMissingError(component="RiskManager", dependency="MarketPriceService instance")
        if self.portfolio_manager is None:
            raise DependencyMissingError(component="RiskManager", dependency="PortfolioManager instance")

    def _ensure_portfolio_manager_prerequisites(self) -> None:
        """Ensure all prerequisites for PortfolioManager are met before instantiation."""
        if self.market_price_service is None:
            raise DependencyMissingError(component="PortfolioManager", dependency="MarketPriceService instance")
        if PortfolioManager is None:  # This is the class itself
            raise DependencyMissingError(component="PortfolioManager", dependency="PortfolioManager class")

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

    def _raise_dependency_not_instantiated(self, component_name: str, dependency_name: str) -> None:
        """Raise DependencyMissingError for a non-instantiated dependency."""
        raise DependencyMissingError(
            component=component_name,
            dependency=f"{dependency_name} not instantiated or available",
        )

    def _raise_logger_service_instantiation_failed(self) -> None:
        """Raise LoggerServiceInstantiationFailedExit."""
        raise LoggerServiceInstantiationFailedExit

    def _raise_service_director_not_initialized(self) -> None:
        """Raise DependencyMissingError for uninitialized ServiceInitializationDirector."""
        raise DependencyMissingError(
            component="LoggerService",
            dependency="ServiceInitializationDirector",
            message="Service director not initialized",
        )

    def _raise_kraken_market_price_service_unavailable_for_live_mode(self) -> None:
        """Raise DependencyMissingError for unavailable KrakenMarketPriceService in live mode."""
        raise DependencyMissingError(
            component="MarketPriceService (live mode)",
            dependency="KrakenMarketPriceService class",
        )

    def _raise_market_price_service_unsupported_mode(self, mode: str) -> None:
        """Raise MarketPriceServiceUnsupportedModeError."""
        # Note: supported_modes can be dynamically fetched or hardcoded if static
        # Get supported modes from configuration instead of hardcoded examples
        if hasattr(self, "mode_manager") and self.mode_manager:
            supported_modes = [mode.value for mode in self.mode_manager.get_supported_modes()]
        else:
            # Fallback if mode manager not available
            supported_modes = ["live_trading", "paper_trading"]

        raise MarketPriceServiceUnsupportedModeError(mode=mode, supported_modes=supported_modes)

    def _raise_market_price_service_critical_failure(self) -> None:
        """Raise MarketPriceServiceCriticalFailureExit."""
        raise MarketPriceServiceCriticalFailureExit

    def _raise_portfolio_manager_instantiation_failed(self) -> None:
        """Raise PortfolioManagerInstantiationFailedExit."""
        raise PortfolioManagerInstantiationFailedExit

    def _create_kraken_exchange_spec(self) -> ExchangeSpecification:
        """Create ExchangeSpecification for Kraken exchange."""
        return ExchangeSpecification(
            exchange_id="kraken",
            exchange_type=ExchangeType.CRYPTO_EXCHANGE,
            name="Kraken",
            supports_limit_orders=True,
            supports_market_orders=True,
            supports_stop_orders=True,
            supports_bracket_orders=False,
            supports_margin=True,
            maker_fee_bps=Decimal(16),  # 0.16%
            taker_fee_bps=Decimal(26),  # 0.26%
            fee_currency="USD",
            provides_l2_data=True,
            provides_tick_data=False,
            max_market_data_depth=1000,
            max_orders_per_second=200,
            supports_websocket=True,
            supports_private_ws=True,
            typical_latency_ms=50.0,
        )

    def _instantiate_execution_handler(self, run_mode: str) -> _ExecutionHandlerType:
        """Instantiate the correct ExecutionHandler based on the run mode.

        Returns:
        -------
            The instantiated execution handler.

        Raises:
        ------
            ExecutionHandlerInstantiationFailedExit: If instantiation fails after attempting.
            DependencyMissingError: If a required component for the handler is missing.
            UnsupportedModeError: If the run_mode is not supported.
        """
        self.execution_handler = None

        # Validate dependencies
        self._validate_execution_handler_dependencies()

        # Create handler based on mode
        handler = self._create_handler_for_mode(run_mode)

        # Finalize handler setup
        return self._finalize_handler_setup(handler, run_mode)

    def _validate_execution_handler_dependencies(self) -> None:
        """Validate that all required dependencies are available."""
        dependency_checks = [
            (self.config is None or ConfigManager is None, "ExecutionHandler", "Config"),
            (self.pubsub is None or PubSubManager is None, "ExecutionHandler", "PubSub"),
            (self.logger_service is None or LoggerService is None, "ExecutionHandler", "LoggerService"),
            (self.monitoring_service is None or MonitoringService is None, "ExecutionHandler", "MonitoringService"),
        ]

        for condition, component, dependency in dependency_checks:
            if condition:
                raise DependencyMissingError(component=component, dependency=dependency)

    def _create_handler_for_mode(self, run_mode: str) -> Any:
        """Create appropriate handler based on run mode."""
        if run_mode in {"live", "live_trading"}:
            return self._create_live_handler()
        if run_mode in {"paper", "paper_trading"}:
            return self._create_paper_handler()
        self._raise_unsupported_mode_error(run_mode)
        return None  # This line is never reached, but satisfies the type checker

    def _create_live_handler(self) -> Any:
        """Create live trading execution handler."""
        if KrakenExecutionHandler is None:
            raise DependencyMissingError(
                component="Live mode ExecutionHandler",
                dependency="KrakenExecutionHandler class",
            )

        # Type-safe dependency validation
        config = DependencyValidator.assert_config_manager(self.config)
        pubsub = DependencyValidator.assert_pubsub_manager(self.pubsub)
        logger = DependencyValidator.assert_logger_service(self.logger_service)
        monitoring = DependencyValidator.assert_monitoring_service(self.monitoring_service)

        handler = KrakenExecutionHandler(
            exchange_spec=self._create_kraken_exchange_spec(),
            config_manager=config,
            pubsub_manager=pubsub,
            logger_service=logger,
            monitoring_service=monitoring,
        )
        log.debug("KrakenExecutionHandler instantiated.")
        return handler

    def _create_paper_handler(self) -> Any:
        """Create paper trading execution handler."""
        if SimulatedExecutionHandler is None:
            raise DependencyMissingError(
                component="Paper mode ExecutionHandler",
                dependency="SimulatedExecutionHandler class",
            )

        if self.historical_data_service is None or HistoricalDataService is None:
            raise DependencyMissingError(
                component="SimulatedExecutionHandler",
                dependency="HistoricalDataService",
            )

        # Type-safe dependency validation
        config = DependencyValidator.assert_config_manager(self.config)
        pubsub = DependencyValidator.assert_pubsub_manager(self.pubsub)
        logger = DependencyValidator.assert_logger_service(self.logger_service)

        handler = SimulatedExecutionHandler(
            config_manager=config,
            pubsub_manager=pubsub,
            data_service=self.historical_data_service,  # type: ignore[arg-type]
            logger_service=logger,
        )
        log.debug("SimulatedExecutionHandler instantiated for paper mode.")
        return handler

    def _raise_unsupported_mode_error(self, run_mode: str) -> None:
        """Raise error for unsupported run mode."""
        if self.mode_manager:
            supported_modes = [mode.value for mode in self.mode_manager.get_supported_modes()]
        else:
            supported_modes = ["live_trading", "paper_trading"]
        raise UnsupportedModeError(mode=run_mode, supported_modes=supported_modes)

    def _finalize_handler_setup(self, handler: Any, run_mode: str) -> _ExecutionHandlerType:
        """Finalize handler setup and add to services."""
        if handler is None:
            log.critical("Execution handler is unexpectedly None after instantiation attempt for mode: %s.", run_mode)
            raise ExecutionHandlerInstantiationFailedExit(mode=run_mode)

        self.execution_handler = handler
        self.services.append(handler)
        return cast("_ExecutionHandlerType", handler)

    async def initialize(self, args: argparse.Namespace) -> None:  # Add return type
        """Load configuration, set up logging, and instantiate components."""
        log.info("Initializing GalFridayApp (Version: %s)...", __version__)
        self.args = args  # args should be guaranteed by main_async

        # Load configuration
        self._load_configuration(args.config)

        # Set up logging
        await self._setup_application_logging(args)

        # Initialize operational mode
        await self._initialize_operational_mode(args)

        # Set up core infrastructure
        self._setup_executor()
        self._instantiate_pubsub()

        # Initialize enterprise dependency injection container
        await self._initialize_service_container()

        # Initialize database components
        await self._initialize_database_components()

        # Initialize logger service with DB support
        await self._initialize_logger_service()

        # Run database migrations
        await self._run_database_migrations()

        # Initialize other services using container
        await self._init_services_with_container()

        log.info("Initialization phase complete.")

    async def _setup_application_logging(self, args: argparse.Namespace) -> None:
        """Set up application logging with enterprise or basic configuration."""
        try:
            # Initialize enterprise logging setup
            config = DependencyValidator.assert_config_manager(self.config)
            self.logging_setup = LoggingSetup(config)
            self.logging_setup.setup_logging(
                log_level=args.log_level,
                log_file=args.log_file,
                enable_json=args.enable_json_logs,
                enable_console=not args.disable_console_logs,
            )
            log.info("Enterprise logging configured successfully.")
        except Exception:
            log.exception("ERROR: Failed to configure enterprise logging")
            # Fallback to basic logging
            setup_logging(self.config, args.log_level)
            log.info("Fallback to basic logging configuration.")

    async def _initialize_operational_mode(self, args: argparse.Namespace) -> None:
        """Initialize the operational mode manager and select mode."""
        try:
            # Initialize mode manager
            config = DependencyValidator.assert_config_manager(self.config)
            self.mode_manager = OperationalModeManager(config)

            # Select and validate operational mode
            self.current_operational_mode = await self.mode_manager.select_and_initialize_operational_mode(
                explicit_mode=args.mode,
            )
            log.info(f"Operational mode initialized: {self.current_operational_mode.value}")

        except Exception:
            log.exception("ERROR: Failed to initialize operational mode")
            # Fallback to safe default
            self.current_operational_mode = OperationalMode.PAPER_TRADING
            log.warning(f"Using fallback operational mode: {self.current_operational_mode.value}")

    async def _initialize_service_container(self) -> None:
        """Initialize the enterprise dependency injection container using type-safe patterns."""
        try:
            # Validate that ConfigManager is available with enhanced type guards
            config = ensure_config_manager(self.config)

            # Use type-safe dependency validation
            DependencyTypeGuards.validate_config_dependency(
                config,
                required_sections=["logging", "database", "operational_modes"],
            )

            # Create and configure the service container
            self.container = create_application_container(config)

            # Validate the container with type safety
            self.container = ensure_service_container(self.container)

            # Initialize the service director
            self.service_director = ServiceInitializationDirector(self.container)
            self.service_director.register_standard_builders()

            log.info("Enterprise dependency injection container initialized successfully")

        except Exception as e:
            log.exception("ERROR: Failed to initialize service container")
            raise DependencyMissingError(
                component="Application",
                dependency="ServiceContainer",
                message=f"Failed to initialize service container: {e}",
            ) from e

    async def _initialize_database_components(self) -> None:
        """Initialize database connection pool and session maker."""
        if DatabaseConnectionPool is None or self.config is None:
            log.critical("DatabaseConnectionPool or its dependencies (ConfigManager) are missing.")
            raise DependencyMissingError("Application", "DatabaseConnectionPool or ConfigManager")

        temp_db_logger = logging.getLogger("gal_friday.db_pool_init")
        self.db_connection_pool = DatabaseConnectionPool(
            config=self.config,
            logger=temp_db_logger,  # type: ignore[arg-type] # Pass a basic logger
        )
        await self.db_connection_pool.initialize()

        self.session_maker = self.db_connection_pool.get_session_maker()
        if not self.session_maker:
            log.critical(
                "Failed to get session_maker from DatabaseConnectionPool. DB-dependent services will fail.",
            )
            raise DependencyMissingError(
                "Application",
                "session_maker from DatabaseConnectionPool",
            )
        log.info("DatabaseConnectionPool initialized and session_maker created.")

    async def _initialize_logger_service(self) -> None:
        """Initialize the logger service using type-safe initialization patterns."""
        if LoggerService is None:
            # mypy doesn't understand runtime import patterns
            self._raise_logger_service_instantiation_failed()  # type: ignore[unreachable]

        try:
            # Use type-safe service initialization
            if self.service_director is None:
                self._raise_service_director_not_initialized()

            # Initialize logger service using the director
            if self.service_director is None:
                self._raise_service_director_not_initialized()
            # Type narrowing: we validated above that service_director is not None
            service_director = cast("ServiceInitializationDirector", self.service_director)
            self.logger_service = await service_director.initialize_service("LoggerService")

            # Note: LoggerService database session maker is set during initialization
            # No need to update it here as it's already configured

            # Validate the logger service with type guards
            self.logger_service = TypeSafeAssertion.assert_logger_service(
                self.logger_service,
                "LoggerService initialization",
            )

            self.services.append(self.logger_service)  # Add to services for start/stop
            log.info("LoggerService instantiated/configured with type safety.")

        except Exception as e:
            log.exception("FATAL: Failed to instantiate full LoggerService")
            raise LoggerServiceInstantiationFailedExit from e

    async def _run_database_migrations(self) -> None:
        """Run database migrations to head."""
        if MigrationManager is None or self.logger_service is None:
            # mypy doesn't understand runtime import patterns
            log.critical("MigrationManager or LoggerService missing, cannot run migrations.")
            raise DependencyMissingError("Application", "MigrationManager or LoggerService")

        self.migration_manager = MigrationManager(
            logger=self.logger_service,  # Pass the full logger service
            project_root_path="/app",  # Explicitly set project root
        )
        log.info("MigrationManager instantiated.")

        try:
            log.info("Running database migrations to head...")
            await asyncio.to_thread(self.migration_manager.upgrade_to_head)
            log.info("Database migrations completed.")
        except Exception:
            log.exception("Failed to run database migrations.")
            raise  # Re-raise as this is critical for app consistency

    async def _init_services_with_container(self) -> None:
        """Initialize core application services using type-safe patterns and dependency injection."""
        container = ensure_service_container(self.container)
        session_maker: async_sessionmaker[AsyncSession] = ensure_not_none(self.session_maker, "session_maker")
        service_director = ensure_not_none(self.service_director, "service_director")

        try:
            # Initialize services using type-safe patterns
            log.info("Initializing services using type-safe patterns...")

            # Use service director to initialize core services
            async with service_director.service_lifecycle() as services:
                # Extract initialized services
                if "MonitoringService" in services:
                    self.monitoring_service = TypeSafeAssertion.assert_monitoring_service(
                        services["MonitoringService"],
                        "MonitoringService from director",
                    )
                    # Note: MonitoringService doesn't implement ServiceProtocol fully (missing initialize method)
                    # so we don't add it to services list for now

                if "CLIService" in services:
                    self.cli_service = TypeSafeAssertion.assert_cli_service(
                        services["CLIService"],
                        "CLIService from director",
                    )
                    self.services.append(self.cli_service)

            # Resolve remaining services from container with type safety
            await self._resolve_remaining_services_safely(container)

            # Use orchestrator for services that need manual setup
            await self._init_manual_services_safely(session_maker)

            log.info(f"Successfully initialized {len(self.services)} services with type safety")

        except Exception:
            log.exception("ERROR: Failed to initialize services with type-safe patterns")
            # Fallback to manual service initialization
            log.warning("Falling back to manual service initialization")
            await self._init_services_fallback()

    async def _resolve_remaining_services_safely(self, container: ServiceContainer) -> None:
        """Resolve remaining services from container with type safety."""
        try:
            # Resolve PubSubManager
            pubsub_manager_type = container.get_type("gal_friday.core.pubsub", "PubSubManager")
            if container.is_registered(pubsub_manager_type):
                self.pubsub = await container.resolve(pubsub_manager_type)
                self.pubsub = TypeSafeAssertion.assert_pubsub_manager(
                    self.pubsub,
                    "PubSubManager from container",
                )

            # Resolve StrategyArbitrator
            strategy_arbitrator_type = container.get_type(
                "gal_friday.strategy_arbitrator",
                "StrategyArbitrator",
            )
            if container.is_registered(strategy_arbitrator_type):
                self.strategy_arbitrator = await container.resolve(strategy_arbitrator_type)
                # Only add to services if it's not None and implements ServiceProtocol
                if self.strategy_arbitrator is not None:
                    self.services.append(cast("ServiceProtocol", self.strategy_arbitrator))

        except Exception as e:
            log.warning(f"Failed to resolve some services from container: {e}")

    async def _init_manual_services_safely(self, session_maker: Any) -> None:
        """Initialize services that need manual setup with type safety."""
        try:
            # Initialize services that still need manual setup
            manual_services: list[Any] = []

            # Only add services that weren't initialized through the container
            if self.strategy_arbitrator is None:
                strategy_arbitrator = self._init_strategy_arbitrator()
                if strategy_arbitrator:
                    manual_services.append(strategy_arbitrator)

            if self.cli_service is None:
                cli_service = self._init_cli_service()
                if cli_service:
                    manual_services.append(cli_service)

            # Use orchestrator for manual services
            if manual_services:
                orchestrator = ServiceOrchestrator(manual_services)
                await orchestrator.initialize_all(session_maker)
                await orchestrator.start_all()
                self.services.extend(orchestrator.services)

        except Exception as e:
            log.warning(f"Failed to initialize some manual services: {e}")

    async def _init_services_fallback(self) -> None:
        """Fallback method for service initialization without container."""
        if self.session_maker is None:
            raise DependencyMissingError(
                component="Service initialization",
                dependency="session_maker",
            )

        orchestrator = ServiceOrchestrator(
            [svc for svc in [self._init_strategy_arbitrator(), self._init_cli_service()] if svc],
        )
        await orchestrator.initialize_all(self.session_maker)
        await orchestrator.start_all()
        self.services.extend(orchestrator.services)

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
                    log.exception("Error creating start task for %s", service_name)
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
        return list[Any](results)  # Ensure we return a list[Any]

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
                log.error("Service task %s failed during startup: %s", task_name, result)
                failed_services.append(task_name)
            else:
                log.info("Service task %s completed startup successfully.", task_name)

        if failed_services:
            log.critical("Critical services failed to start: %s. Initiating shutdown.", ", ".join(failed_services))
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
        if self._config_manager_instance and hasattr(self._config_manager_instance, "start_watching"):
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
                log.debug("Task[Any] %s cancelled successfully.", task_name)
            elif isinstance(result, Exception):
                error_count += 1
                log.error("Error during cancellation/completion of task %s: %s", task_name, result)
        log.info("Service task cancellation complete. Cancelled: %s, Errors: %s", cancelled_count, error_count)
        self.running_tasks.clear()

    async def _shutdown_process_executor(self) -> None:
        """Shuts down the ProcessPoolExecutor."""
        if self.executor:
            log.info("Shutting down ProcessPoolExecutor...")
            try:
                loop = asyncio.get_running_loop()
                # Ensure shutdown is non-blocking in the main async flow
                await loop.run_in_executor(
                    None,
                    functools.partial(self.executor.shutdown, wait=True, cancel_futures=True),
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
        if self._config_manager_instance and hasattr(self._config_manager_instance, "stop_watching"):
            log.info("Stopping configuration file watcher...")
            self._config_manager_instance.stop_watching()

        # 1. Stop services and PubSubManager
        await self._initiate_service_shutdown()

        # 2. Cancel any running tasks created during start()
        await self._cancel_active_tasks()

        # 3. Close DatabaseConnectionPool
        if self.db_connection_pool:
            log.info("Closing DatabaseConnectionPool...")
            await self.db_connection_pool.close()
            log.info("DatabaseConnectionPool closed.")

        # 4. Shutdown the executor
        await self._shutdown_process_executor()

        # 5. Clean up initialization manager
        if self.initialization_manager:
            log.info("Cleaning up initialization manager...")
            await self.initialization_manager.cleanup_all()
            self.initialization_manager = None
            log.info("Initialization manager cleaned up.")

        # 6. Dispose of the service container
        if self.container:
            log.info("Disposing of service container...")
            self.container.dispose()
            self.container = None
            log.info("Service container disposed.")

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

    def _raise_pubsub_error(self) -> None:
        """Helper method to raise PubSubManagerInstantiationFailedExit, abstracting the raise from try block."""
        raise PubSubManagerInstantiationFailedExit


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


# --- Main Entry Point --- #
def main() -> None:
    """Main entry point for the Gal Friday trading system.

    This function provides a production-ready entry point that:
    1. Handles command line argument parsing with comprehensive validation
    2. Sets up enterprise-grade logging with multiple handlers
    3. Implements operational mode selection and validation
    4. Provides configuration-driven mode management
    5. Includes proper error handling and graceful shutdown
    """
    try:
        # Parse command line arguments
        args = _parse_command_line_args()
        if args is None:
            return  # Help or version was displayed

        # Handle special modes
        if _handle_special_modes(args):
            return  # Special mode was handled

        # Normal application startup
        asyncio.run(main_async(args))

    except KeyboardInterrupt:
        sys.exit(0)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)


def _parse_command_line_args() -> argparse.Namespace | None:
    """Parse command line arguments and handle help/version requests."""
    # Quick check for help/version requests
    if len(sys.argv) > 1 and (sys.argv[1] in ["--help", "-h", "--version", "-V"]):
        cli_parser = EnhancedCLIParser()
        cli_parser.parse_args()
        return None

    # Parse arguments
    cli_parser = EnhancedCLIParser()
    return cli_parser.parse_args()


def _handle_special_modes(args: argparse.Namespace) -> bool:
    """Handle special modes like config validation and health check.

    Returns:
        True if a special mode was handled, False otherwise.
    """
    if args.validate_config:
        _handle_config_validation(args)
        return True

    if args.health_check:
        _handle_health_check(args)
        return True

    return False


def _handle_config_validation(args: argparse.Namespace) -> None:
    """Handle configuration validation mode."""
    try:
        if not Path(args.config).exists():
            sys.exit(1)

        # Create temporary config manager for validation
        temp_config = ConfigManager(
            config_path=args.config,
            logger_service=logging.getLogger("config_validator"),
        )

        if temp_config.is_valid():
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception:
        sys.exit(1)


def _handle_health_check(args: argparse.Namespace) -> None:
    """Handle health check mode."""
    try:
        health_status = _perform_health_checks(args)

        if all(health_status.values()):
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception:
        sys.exit(1)


def _perform_health_checks(args: argparse.Namespace) -> dict[str, bool]:
    """Perform health checks and return status."""
    health_status = {
        "config_file_exists": Path(args.config).exists(),
        "log_directory_writable": True,
        "python_version_compatible": sys.version_info >= (3, 8),
    }

    if args.log_file:
        health_status["log_directory_writable"] = _check_log_directory_writable(args.log_file)

    return health_status


def _check_log_directory_writable(log_file: str) -> bool:
    """Check if log directory is writable."""
    try:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        test_file = log_dir / "health_check_test.tmp"
        test_file.touch()
        test_file.unlink()
    except Exception:
        return False
    else:
        return True


if __name__ == "__main__":
    main()
