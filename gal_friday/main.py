#!/usr/bin/env python
"""Main entry point for the Gal-Friday trading bot application.

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
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path  # PTHxxx fix: Import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

# Import exchange specification classes
from .config_manager import ConfigManager as ConfigManagerType
from .core.asset_registry import ExchangeSpecification, ExchangeType

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
    validation_checks: Dict[str, bool]
    warnings: List[str]
    errors: List[str]
    missing_requirements: List[str]


class ModeConfiguration(BaseModel):
    """Configuration for operational modes."""
    
    supported_modes: List[OperationalMode] = Field(
        description="List of supported operational modes"
    )
    
    default_mode: OperationalMode = Field(
        default=OperationalMode.PAPER_TRADING,
        description="Default mode if none specified"
    )
    
    mode_capabilities: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Capabilities required for each mode"
    )
    
    incompatible_combinations: List[List[OperationalMode]] = Field(
        default_factory=list,
        description="Mode combinations that are not allowed"
    )
    
    environment_restrictions: Dict[str, List[OperationalMode]] = Field(
        default_factory=dict,
        description="Mode restrictions per environment (dev, staging, prod)"
    )
    
    @field_validator('supported_modes')
    @classmethod
    def validate_supported_modes(cls, v):
        """Validate that supported modes are valid and non-empty."""
        if not v:
            raise ValueError("At least one operational mode must be supported")
        
        # Ensure all modes are valid enum values
        valid_modes = set(OperationalMode)
        for mode in v:
            if mode not in valid_modes:
                raise ValueError(f"Invalid mode: {mode}")
        
        return v
    
    @field_validator('default_mode')
    @classmethod 
    def validate_default_mode(cls, v, info):
        """Validate that default mode is in supported modes."""
        if info.data and 'supported_modes' in info.data:
            supported_modes = info.data['supported_modes']
            if v not in supported_modes:
                raise ValueError("Default mode must be in supported modes list")
        return v


class OperationalModeManager:
    """Enterprise-grade operational mode management system."""
    
    def __init__(self, config_manager: ConfigManagerType):
        """Initialize the mode manager with configuration."""
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.mode_config = self._load_mode_configuration()
        self.current_mode: Optional[OperationalMode] = None
        self.current_environment: Optional[EnvironmentType] = None
        
        # Define default capabilities for each mode
        self.default_capabilities = {
            OperationalMode.LIVE_TRADING: ModeCapabilities(
                requires_market_data=True,
                requires_execution_handler=True,
                requires_portfolio_manager=True,
                requires_risk_manager=True,
                requires_prediction_service=True,
                supports_real_money=True,
                resource_intensity='high',
                description='Full live trading with real money'
            ),
            OperationalMode.PAPER_TRADING: ModeCapabilities(
                requires_market_data=True,
                requires_execution_handler=True,
                requires_portfolio_manager=True,
                requires_risk_manager=True,
                requires_prediction_service=True,
                supports_real_money=False,
                resource_intensity='medium',
                description='Simulated trading with fake money'
            ),
            OperationalMode.BACKTESTING: ModeCapabilities(
                requires_market_data=True,
                requires_execution_handler=False,
                requires_portfolio_manager=True,
                requires_risk_manager=True,
                requires_prediction_service=True,
                supports_real_money=False,
                resource_intensity='high',
                description='Historical strategy testing'
            ),
            OperationalMode.DATA_COLLECTION: ModeCapabilities(
                requires_market_data=True,
                requires_execution_handler=False,
                requires_portfolio_manager=False,
                requires_risk_manager=False,
                requires_prediction_service=False,
                supports_real_money=False,
                resource_intensity='low',
                description='Market data collection only'
            ),
            OperationalMode.MONITORING_ONLY: ModeCapabilities(
                requires_market_data=True,
                requires_execution_handler=False,
                requires_portfolio_manager=True,
                requires_risk_manager=True,
                requires_prediction_service=False,
                supports_real_money=False,
                resource_intensity='low',
                description='Monitor positions without trading'
            ),
            OperationalMode.MODEL_TRAINING: ModeCapabilities(
                requires_market_data=True,
                requires_execution_handler=False,
                requires_portfolio_manager=False,
                requires_risk_manager=False,
                requires_prediction_service=True,
                supports_real_money=False,
                resource_intensity='high',
                description='Train predictive models'
            ),
            OperationalMode.RESEARCH: ModeCapabilities(
                requires_market_data=True,
                requires_execution_handler=False,
                requires_portfolio_manager=False,
                requires_risk_manager=False,
                requires_prediction_service=False,
                supports_real_money=False,
                resource_intensity='medium',
                description='Research and analysis mode'
            ),
        }
    
    def get_supported_modes(self) -> List[OperationalMode]:
        """Get supported modes from configuration, replacing hardcoded examples.
        
        This replaces line 709: supported_modes=["live", "paper"] with configuration-driven values.
        """
        try:
            # Load from configuration
            supported_modes = self.mode_config.supported_modes
            
            # Apply environment restrictions
            current_env = self.config_manager.get('environment', 'development')
            if current_env in self.mode_config.environment_restrictions:
                allowed_modes = self.mode_config.environment_restrictions[current_env]
                supported_modes = [mode for mode in supported_modes if mode in allowed_modes]
            
            self.logger.info(
                f"Loaded {len(supported_modes)} supported modes from configuration: "
                f"{[mode.value for mode in supported_modes]}"
            )
            
            return supported_modes
            
        except Exception as e:
            self.logger.error(f"Error loading supported modes from configuration: {e}")
            # Fallback to safe default
            fallback_modes = [OperationalMode.PAPER_TRADING, OperationalMode.DATA_COLLECTION]
            self.logger.warning(f"Using fallback modes: {[mode.value for mode in fallback_modes]}")
            return fallback_modes
    
    def detect_environment(self) -> EnvironmentType:
        """Detect current environment based on various indicators."""
        # Check environment variable first
        env_var = os.getenv('GAL_FRIDAY_ENV', '').lower()
        if env_var:
            try:
                return EnvironmentType(env_var)
            except ValueError:
                self.logger.warning(f"Invalid environment variable value: {env_var}")
        
        # Check configuration file
        config_env = self.config_manager.get('environment', '').lower()
        if config_env:
            try:
                return EnvironmentType(config_env)
            except ValueError:
                self.logger.warning(f"Invalid environment in config: {config_env}")
        
        # Auto-detect based on system characteristics
        if self._is_production_environment():
            return EnvironmentType.PRODUCTION
        elif self._is_staging_environment():
            return EnvironmentType.STAGING
        elif self._is_development_environment():
            return EnvironmentType.DEVELOPMENT
        else:
            return EnvironmentType.LOCAL
    
    async def select_and_initialize_operational_mode(self, explicit_mode: Optional[str] = None) -> OperationalMode:
        """Select and initialize the appropriate operational mode.
        
        This implements the TODO at line 922: operational mode selection logic.
        """
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
                raise ValueError(error_msg)
            
            # Log warnings if any
            for warning in validation_result.warnings:
                self.logger.warning(f"Mode validation warning: {warning}")
            
            # Set current mode and environment
            self.current_mode = mode
            self.current_environment = environment
            
            self.logger.info(
                f"Operational mode initialization complete: {mode.value} in {environment.value} environment"
            )
            
            return mode
            
        except Exception as e:
            self.logger.error(f"Error in operational mode selection: {e}")
            raise
    
    async def _determine_operational_mode(
        self, 
        environment: EnvironmentType,
        explicit_mode: Optional[str] = None
    ) -> OperationalMode:
        """Determine operational mode based on environment and configuration."""
        # Check explicit mode configuration first
        if explicit_mode:
            try:
                mode = OperationalMode(explicit_mode)
                self.logger.debug(f"Using explicit mode configuration: {mode.value}")
                return mode
            except ValueError:
                self.logger.warning(f"Invalid explicit mode configuration: {explicit_mode}")
        
        # Check configuration file mode
        config_mode = self.config_manager.get('operational_mode', '').lower()
        if config_mode:
            try:
                mode = OperationalMode(config_mode)
                self.logger.debug(f"Using configuration file mode: {mode.value}")
                return mode
            except ValueError:
                self.logger.warning(f"Invalid mode in configuration: {config_mode}")
        
        # Environment-based defaults
        if environment == EnvironmentType.PRODUCTION:
            return OperationalMode.LIVE_TRADING
        elif environment == EnvironmentType.STAGING:
            return OperationalMode.PAPER_TRADING
        elif environment == EnvironmentType.DEVELOPMENT:
            return OperationalMode.PAPER_TRADING
        else:  # LOCAL
            return OperationalMode.BACKTESTING
    
    async def _validate_mode_compatibility(
        self, 
        mode: OperationalMode, 
        environment: EnvironmentType
    ) -> ModeValidationResult:
        """Validate that the selected mode is compatible with current environment."""
        validation_checks = {}
        warnings = []
        errors = []
        missing_requirements = []
        
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
                missing_requirements=missing_requirements
            )
        
        # Validate environment-specific restrictions
        if environment == EnvironmentType.PRODUCTION and not capabilities.supports_real_money:
            warnings.append("Non-money mode in production environment")
        
        # Basic capability checks (simplified - would need actual service availability checks)
        validation_checks['market_data'] = capabilities.requires_market_data
        validation_checks['execution_handler'] = capabilities.requires_execution_handler
        validation_checks['portfolio_manager'] = capabilities.requires_portfolio_manager
        validation_checks['risk_manager'] = capabilities.requires_risk_manager
        validation_checks['prediction_service'] = capabilities.requires_prediction_service
        
        is_valid = len(errors) == 0
        
        return ModeValidationResult(
            is_valid=is_valid,
            mode=mode,
            environment=environment,
            validation_checks=validation_checks,
            warnings=warnings,
            errors=errors,
            missing_requirements=missing_requirements
        )
    
    def validate_mode_request(self, requested_modes: List[str]) -> List[OperationalMode]:
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
                    raise ValueError(f"Mode '{mode_str}' is not supported in current configuration")
                
                validated_modes.append(mode)
                
            except ValueError as e:
                available_modes = [mode.value for mode in supported_modes]
                raise ValueError(
                    f"Invalid mode '{mode_str}'. Available modes: {available_modes}"
                ) from e
        
        # Check for incompatible combinations
        self._validate_mode_compatibility_rules(validated_modes)
        
        return validated_modes
    
    def _validate_mode_compatibility_rules(self, modes: List[OperationalMode]) -> None:
        """Check for incompatible mode combinations."""
        mode_set = set(modes)
        
        # Check against configured incompatible combinations
        for incompatible_group in self.mode_config.incompatible_combinations:
            if len(mode_set.intersection(set(incompatible_group))) > 1:
                conflicting = mode_set.intersection(set(incompatible_group))
                raise ValueError(
                    f"Incompatible mode combination detected: {[m.value for m in conflicting]}"
                )
        
        # Built-in compatibility checks
        if (OperationalMode.LIVE_TRADING in mode_set and 
            OperationalMode.BACKTESTING in mode_set):
            raise ValueError("Cannot run live trading and backtesting simultaneously")
        
        if (OperationalMode.LIVE_TRADING in mode_set and 
            OperationalMode.PAPER_TRADING in mode_set):
            raise ValueError("Cannot run live trading and paper trading simultaneously")
    
    def get_required_services(self, modes: List[OperationalMode]) -> Dict[str, bool]:
        """Determine which services are required for the given modes."""
        requirements = {
            'market_data': False,
            'execution_handler': False,
            'portfolio_manager': False,
            'risk_manager': False,
            'prediction_service': False
        }
        
        for mode in modes:
            capabilities = self.default_capabilities.get(mode)
            if capabilities:
                requirements['market_data'] |= capabilities.requires_market_data
                requirements['execution_handler'] |= capabilities.requires_execution_handler
                requirements['portfolio_manager'] |= capabilities.requires_portfolio_manager
                requirements['risk_manager'] |= capabilities.requires_risk_manager
                requirements['prediction_service'] |= capabilities.requires_prediction_service
        
        return requirements
    
    def _load_mode_configuration(self) -> ModeConfiguration:
        """Load mode configuration from ConfigManager."""
        try:
            mode_config_dict = self.config_manager.get('operational_modes', {})
            
            # Provide defaults if configuration is incomplete
            if not mode_config_dict.get('supported_modes'):
                mode_config_dict['supported_modes'] = [
                    OperationalMode.PAPER_TRADING,
                    OperationalMode.BACKTESTING,
                    OperationalMode.DATA_COLLECTION
                ]
            
            return ModeConfiguration(**mode_config_dict)
            
        except Exception as e:
            self.logger.error(f"Error loading mode configuration: {e}")
            # Return safe default configuration
            return ModeConfiguration(
                supported_modes=[
                    OperationalMode.PAPER_TRADING,
                    OperationalMode.DATA_COLLECTION
                ],
                default_mode=OperationalMode.PAPER_TRADING
            )
    
    def _is_production_environment(self) -> bool:
        """Check if running in production environment."""
        production_indicators = [
            os.getenv('IS_PRODUCTION', '').lower() == 'true',
            os.path.exists('/etc/gal-friday/production.conf'),
            'production' in os.getcwd().lower(),
            os.getenv('DEPLOYMENT_ENV', '').lower() == 'production'
        ]
        return any(production_indicators)
    
    def _is_staging_environment(self) -> bool:
        """Check if running in staging environment."""
        staging_indicators = [
            os.getenv('IS_STAGING', '').lower() == 'true',
            os.path.exists('/etc/gal-friday/staging.conf'),
            'staging' in os.getcwd().lower(),
            os.getenv('DEPLOYMENT_ENV', '').lower() == 'staging'
        ]
        return any(staging_indicators)
    
    def _is_development_environment(self) -> bool:
        """Check if running in development environment."""
        development_indicators = [
            os.getenv('IS_DEVELOPMENT', '').lower() == 'true',
            os.path.exists('pyproject.toml'),  # Development workspace indicator
            '.git' in os.listdir('.') if os.path.exists('.') else False,
            os.getenv('DEPLOYMENT_ENV', '').lower() == 'development'
        ]
        return any(development_indicators)


# --- Enhanced Logging Setup System --- #
class LoggingSetup:
    """Enterprise-grade logging configuration and initialization.
    
    This implements the TODO at line 923: logging setup and CLI initialization.
    """
    
    def __init__(self, config: ConfigManagerType):
        """Initialize logging setup with configuration."""
        self.config = config
        self.logger = None
    
    def setup_logging(
        self, 
        log_level: Optional[str] = None,
        log_file: Optional[str] = None,
        enable_json: bool = False,
        enable_console: bool = True
    ) -> logging.Logger:
        """Setup comprehensive logging system with multiple handlers and formatters."""
        # Determine log level
        if log_level:
            level = getattr(logging, log_level.upper(), logging.INFO)
        else:
            level = getattr(logging, self.config.get('logging', {}).get('level', 'INFO').upper(), logging.INFO)
        
        # Create formatters
        formatters = self._create_formatters(enable_json)
        
        # Create handlers
        handlers = self._create_handlers(log_file, enable_console, formatters)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            handlers=handlers,
            force=True  # Override any existing configuration
        )
        
        # Configure specific loggers
        self._configure_library_loggers()
        
        # Create application logger
        logger = logging.getLogger('gal_friday')
        logger.setLevel(level)
        
        # Log initial startup information
        logger.info(f"Logging initialized - Level: {logging.getLevelName(level)}")
        logger.info(f"Log handlers: {[type(h).__name__ for h in handlers]}")
        
        self.logger = logger
        return logger
    
    def _create_formatters(self, enable_json: bool) -> Dict[str, logging.Formatter]:
        """Create logging formatters."""
        formatters = {}
        
        # Standard formatter
        formatters['standard'] = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Detailed formatter
        formatters['detailed'] = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # JSON formatter (simplified implementation)
        if enable_json:
            formatters['json'] = logging.Formatter(
                '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        return formatters
    
    def _create_handlers(
        self, 
        log_file: Optional[str],
        enable_console: bool,
        formatters: Dict[str, logging.Formatter]
    ) -> List[logging.Handler]:
        """Create logging handlers."""
        handlers = []
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatters['standard'])
            handlers.append(console_handler)
        
        # File handler with rotation
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(formatters['detailed'])
            handlers.append(file_handler)
        
        # Error file handler
        if log_file:
            error_log_file = str(log_path.parent / f"{log_path.stem}_errors{log_path.suffix}")
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=3
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatters['detailed'])
            handlers.append(error_handler)
        
        return handlers
    
    def _configure_library_loggers(self) -> None:
        """Configure third-party library loggers to reduce noise."""
        # Reduce verbosity of common libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.INFO)


# --- Enhanced CLI Parser System --- #
class EnhancedCLIParser:
    """Command line interface parser with comprehensive options.
    
    This enhances the basic argparse setup for the TODO at line 923.
    """
    
    def __init__(self):
        """Initialize the CLI parser."""
        self.parser = None
        self._setup_parser()
    
    def _setup_parser(self) -> None:
        """Setup command line argument parser with comprehensive options."""
        self.parser = argparse.ArgumentParser(
            description='Gal Friday - Enterprise Trading System',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s --mode live_trading --config config/production.yaml
  %(prog)s --mode paper_trading --log-level DEBUG
  %(prog)s --mode backtesting --log-file logs/backtest.log
  %(prog)s --validate-config --config config/staging.yaml
            """
        )
        
        # Operational mode arguments
        mode_group = self.parser.add_argument_group('Operational Mode')
        mode_group.add_argument(
            '--mode', '-m',
            choices=[mode.value for mode in OperationalMode],
            default=OperationalMode.PAPER_TRADING.value,
            help='Operational mode (default: paper_trading)'
        )
        
        # Configuration arguments
        config_group = self.parser.add_argument_group('Configuration')
        config_group.add_argument(
            '--config', '-c',
            type=str,
            default='config/default.yaml',
            help='Path to configuration file (default: config/default.yaml)'
        )
        config_group.add_argument(
            '--validate-config',
            action='store_true',
            help='Validate configuration and exit'
        )
        
        # Logging arguments
        log_group = self.parser.add_argument_group('Logging')
        log_group.add_argument(
            '--log-level', '-l',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            help='Set logging level'
        )
        log_group.add_argument(
            '--log-file',
            type=str,
            help='Path to log file'
        )
        log_group.add_argument(
            '--enable-json-logs',
            action='store_true',
            help='Enable JSON structured logging'
        )
        log_group.add_argument(
            '--disable-console-logs',
            action='store_true',
            help='Disable console logging output'
        )
        
        # Environment and deployment
        env_group = self.parser.add_argument_group('Environment')
        env_group.add_argument(
            '--environment',
            choices=[env.value for env in EnvironmentType],
            help='Deployment environment'
        )
        
        # Application control
        control_group = self.parser.add_argument_group('Application Control')
        control_group.add_argument(
            '--dry-run',
            action='store_true',
            help='Perform dry run without executing trades'
        )
        control_group.add_argument(
            '--health-check',
            action='store_true',
            help='Perform health check and exit'
        )
        control_group.add_argument(
            '--version', '-V',
            action='version',
            version=f'Gal Friday Trading System v{__version__}'
        )
    
    def parse_args(self, args=None) -> argparse.Namespace:
        """Parse command line arguments with validation."""
        parsed_args = self.parser.parse_args(args)
        
        # Validate arguments
        self._validate_args(parsed_args)
        
        return parsed_args
    
    def _validate_args(self, parsed_args: argparse.Namespace) -> None:
        """Validate parsed arguments."""
        # Validate config file exists if specified
        if parsed_args.config and not os.path.exists(parsed_args.config):
            if not parsed_args.validate_config:  # Don't fail if just validating
                self.parser.error(f"Configuration file not found: {parsed_args.config}")
        
        # Validate log file directory exists or can be created
        if parsed_args.log_file:
            log_dir = Path(parsed_args.log_file).parent
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                self.parser.error(f"Cannot create log directory {log_dir}: {e}")


def initialize_application_components(config_manager: ConfigManagerType) -> tuple[logging.Logger, argparse.Namespace, OperationalModeManager]:
    """Initialize application with enhanced logging, CLI setup, and mode management.
    
    This is the main entry point that replaces TODOs at lines 709, 922, and 923.
    """
    # Setup command line interface
    cli_parser = EnhancedCLIParser()
    args = cli_parser.parse_args()
    
    # Setup logging
    logging_setup = LoggingSetup(config_manager)
    logger = logging_setup.setup_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        enable_json=args.enable_json_logs,
        enable_console=not args.disable_console_logs
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


# --- Conditional Imports for Type Checking --- #
if TYPE_CHECKING:
    # ERA001: Removed commented-out import.
    # Define a proper protocol/interface for execution handlers
    from typing import Protocol

    from sqlalchemy.ext.asyncio import async_sessionmaker  # Added

    from .cli_service import CLIService as CLIServiceType
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
    startup_logger.error("Failed to import ConfigManager")
    ConfigManager = None  # type: ignore[assignment,misc]

try:
    from .core.pubsub import PubSubManager
except ImportError:
    startup_logger.error("Failed to import PubSubManager")
    PubSubManager = None  # type: ignore[assignment,misc]

try:
    from .data_ingestor import DataIngestor
except ImportError:
    startup_logger.error("Failed to import DataIngestor")
    DataIngestor = None  # type: ignore[assignment,misc]

try:
    from .prediction_service import PredictionService
except ImportError:
    startup_logger.error("Failed to import PredictionService")
    PredictionService = None  # type: ignore[assignment,misc]

try:
    from .strategy_arbitrator import StrategyArbitrator
except ImportError:
    startup_logger.error("Failed to import StrategyArbitrator")
    StrategyArbitrator = None  # type: ignore[assignment,misc]

try:
    from .portfolio_manager import PortfolioManager
except ImportError:
    startup_logger.error("Failed to import PortfolioManager")
    PortfolioManager = None  # type: ignore[assignment,misc]

try:
    from .risk_manager import RiskManager
except ImportError:
    startup_logger.error("Failed to import RiskManager")
    RiskManager = None  # type: ignore[assignment,misc]

# --- Execution Handler Imports (Runtime) --- #
try:
    from .execution.kraken import KrakenExecutionHandler
except ImportError as e:
    startup_logger.error("Failed to import KrakenExecutionHandler: %s", e)
    KrakenExecutionHandler = None  # type: ignore

try:
    from .simulated_execution_handler import SimulatedExecutionHandler
except ImportError:
    startup_logger.error("Failed to import SimulatedExecutionHandler")
    SimulatedExecutionHandler = None  # type: ignore

# --- Other Service Imports (Runtime) --- #
try:
    from .logger_service import LoggerService
except ImportError:
    startup_logger.error("Failed to import LoggerService")
    LoggerService = None  # type: ignore[assignment,misc]

try:
    from .monitoring_service import MonitoringService
except ImportError:
    startup_logger.error("Failed to import MonitoringService")
    MonitoringService = None  # type: ignore[assignment,misc]

try:
    from .cli_service import CLIService
except ImportError:
    startup_logger.error("Failed to import CLIService")
    CLIService = None  # type: ignore[assignment,misc]

try:
    from .market_price_service import MarketPriceService
except ImportError:
    startup_logger.error("Failed to import MarketPriceService")
    MarketPriceService = None  # type: ignore[assignment,misc]

try:
    from .historical_data_service import HistoricalDataService
except ImportError:
    startup_logger.error("Failed to import HistoricalDataService")
    HistoricalDataService = None  # type: ignore[assignment,misc]

try:
    from .core.feature_registry_client import FeatureRegistryClient
except ImportError:
    startup_logger.error("Failed to import FeatureRegistryClient")
    FeatureRegistryClient = None  # type: ignore[assignment,misc]

# --- DAL Imports ---
try:
    from .dal.connection_pool import DatabaseConnectionPool
except ImportError:
    startup_logger.error("Failed to import DatabaseConnectionPool")
    DatabaseConnectionPool = None # type: ignore[assignment,misc]

try:
    from .dal.migrations.migration_manager import MigrationManager
except ImportError:
    startup_logger.error("Failed to import MigrationManager")
    MigrationManager = None # type: ignore[assignment,misc]


# --- Import concrete service implementations --- #
try:
    from .market_price.kraken_service import KrakenMarketPriceService
except ImportError as e:
    startup_logger.warning("Failed to import KrakenMarketPriceService: %s", e)
    KrakenMarketPriceService = None  # type: ignore

try:
    from .kraken_historical_data_service import KrakenHistoricalDataService
except ImportError as e:
    startup_logger.warning("Failed to import KrakenHistoricalDataService: %s", e)
    KrakenHistoricalDataService = None  # type: ignore

try:
    from .simulated_market_price_service import (  # Restored runtime import
        SimulatedMarketPriceService,
    )
except ImportError:
    startup_logger.error("Failed to import SimulatedMarketPriceService")
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
    config: Optional["ConfigManagerType"],
    log_level_override: str | None = None,
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
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        console_handler = logging.StreamHandler(sys.stdout)
        # Handler level defaults to root logger level
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            console_format,
            datefmt=log_config.get("date_format"),
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
                    "format",
                    "%(asctime)s %(name)s %(levelname)s %(message)s",
                )

                file_handler = logging.handlers.RotatingFileHandler(
                    log_filename,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                )
                file_handler.setLevel(log_level)
                file_formatter = logging.Formatter(
                    file_format,
                    datefmt=log_config.get("date_format"),
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

        # Enterprise-grade components
        self.mode_manager: OperationalModeManager | None = None
        self.logging_setup: LoggingSetup | None = None
        self.current_operational_mode: OperationalMode | None = None

        # Store references to specific services after instantiation for DI
        self.logger_service: LoggerServiceType | None = None
        self.db_connection_pool: DatabaseConnectionPool | None = None # Added
        self.session_maker: async_sessionmaker | None = None # type: ignore # Added
        self.migration_manager: MigrationManager | None = None # Added
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

    def _load_configuration(self, config_path: str) -> None:  # Accept config_path parameter
        """Load the application configuration."""
        try:
            self._ensure_class_available(ConfigManager, "ConfigManager", "Configuration loading")
            # Pass pubsub and loop to ConfigManager for dynamic reloading
            # Pubsub might not be initialized yet. Loop should be available from main_async.
            self._config_manager_instance = ConfigManager(
                config_path=config_path,
                logger_service=logging.getLogger(
                    "gal_friday.config_manager",
                ),  # Give it its own logger instance
            )
            self.config = self._config_manager_instance  # Main access point for config
            log.info("Configuration loaded successfully from: %s", config_path)
            if not self.config.is_valid():
                log.error(
                    "Initial configuration is invalid. Review errors logged by ConfigManager.",
                )
                # Depending on desired strictness, could raise ConfigurationLoadingFailedExit here.
        except Exception as e:
            log.exception(
                "FATAL: Failed to load configuration from %s",
                config_path,
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
            "ConfigManager instance not available for PubSubManager. " "Load configuration first."
        )
        log.critical(error_msg)
        raise DependencyMissingError("PubSubManager", error_msg)

    def _update_config_manager_with_pubsub(self) -> None:
        """Update ConfigManager with PubSubManager instance if available."""
        if self._config_manager_instance and hasattr(
            self._config_manager_instance,
            "set_pubsub_manager",
        ):
            self._config_manager_instance.set_pubsub_manager(self.pubsub)
        if hasattr(self, "_logger"):
            self._logger.info("Updated ConfigManager with PubSubManager instance.")

    def _instantiate_pubsub(self) -> None:
        """Instantiate the PubSubManager."""
        try:
            self._ensure_class_available(
                PubSubManager,
                "PubSubManager",
                "PubSubManager instantiation",
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
        # Prerequisites will be checked in this order and fail fast
        self._ensure_strategy_arbitrator_prerequisites()

        if self.config is None:
            raise RuntimeError("Configuration not initialized")

        strategy_arbitrator_config = self.config.get("strategy_arbitrator", {})
        if self.market_price_service is None:
            raise MarketPriceServiceError
        if self.pubsub is None:
            raise PubSubManagerError
        if self.logger_service is None:
            raise LoggerServiceError

        # Create FeatureRegistryClient instance
        if FeatureRegistryClient is not None:
            self.feature_registry_client = FeatureRegistryClient()
            log.debug("FeatureRegistryClient instantiated.")
        else:
            log.warning("FeatureRegistryClient not available, StrategyArbitrator will operate without feature validation.")
            self.feature_registry_client = None

        self.strategy_arbitrator = StrategyArbitrator(
            config=strategy_arbitrator_config,
            pubsub_manager=self.pubsub,
            logger_service=self.logger_service,
            market_price_service=self.market_price_service,
            feature_registry_client=self.feature_registry_client,  # Added parameter
            session_maker=self.session_maker,  # Pass session_maker as a parameter
        )
        log.debug("StrategyArbitrator instantiated.")

    def _init_cli_service(self) -> None:
        """Instantiate the CLIService."""
        if CLIService is not None:
            # Prerequisite checks for CLIService (monitoring_service, portfolio_manager)
            if self.monitoring_service is None:
                raise DependencyMissingError(
                    component="CLIService",
                    dependency="MonitoringService instance",
                )
            if self.portfolio_manager is None:
                raise DependencyMissingError(
                    component="CLIService",
                    dependency="PortfolioManager instance",
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
                component="StrategyArbitrator",
                dependency="MarketPriceService instance",
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
                component="RiskManager",
                dependency="MarketPriceService instance",
            )
        if self.portfolio_manager is None:
            raise DependencyMissingError(
                component="RiskManager",
                dependency="PortfolioManager instance",
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
                component="PortfolioManager",
                dependency="MarketPriceService instance",
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
        self,
        component_name: str,
        dependency_name: str,
    ) -> None:
        """Raise DependencyMissingError for a non-instantiated dependency."""
        raise DependencyMissingError(
            component=component_name,
            dependency=f"{dependency_name} not instantiated or available",
        )

    def _raise_logger_service_instantiation_failed(self) -> None:
        """Raise LoggerServiceInstantiationFailedExit."""
        raise LoggerServiceInstantiationFailedExit

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
        if hasattr(self, 'mode_manager') and self.mode_manager:
            supported_modes = [mode.value for mode in self.mode_manager.get_supported_modes()]
        else:
            # Fallback if mode manager not available
            supported_modes = ["live_trading", "paper_trading"]
        
        raise MarketPriceServiceUnsupportedModeError(
            mode=mode,
            supported_modes=supported_modes,
        )

    def _raise_market_price_service_critical_failure(self) -> None:
        """Raise MarketPriceServiceCriticalFailureExit."""
        raise MarketPriceServiceCriticalFailureExit

    def _raise_portfolio_manager_instantiation_failed(self) -> None:
        """Raise PortfolioManagerInstantiationFailedExit."""
        raise PortfolioManagerInstantiationFailedExit

    def _create_kraken_exchange_spec(self) -> ExchangeSpecification:
        """Create ExchangeSpecification for Kraken exchange."""
        from decimal import Decimal

        return ExchangeSpecification(
            exchange_id="kraken",
            exchange_type=ExchangeType.CRYPTO_EXCHANGE,
            name="Kraken",
            supports_limit_orders=True,
            supports_market_orders=True,
            supports_stop_orders=True,
            supports_bracket_orders=False,
            supports_margin=True,
            maker_fee_bps=Decimal("16"),  # 0.16%
            taker_fee_bps=Decimal("26"),  # 0.26%
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
                component="ExecutionHandler",
                dependency="MonitoringService",
            )

        # Use enterprise operational mode system
        if run_mode == "live" or run_mode == "live_trading":
            if KrakenExecutionHandler is None:
                raise DependencyMissingError(
                    component="Live mode ExecutionHandler",
                    dependency="KrakenExecutionHandler class",
                )
            self.execution_handler = KrakenExecutionHandler(
                exchange_spec=self._create_kraken_exchange_spec(),  # Create ExchangeSpec
                config_manager=self.config,
                pubsub_manager=self.pubsub,
                logger_service=self.logger_service,
                monitoring_service=self.monitoring_service,
            )
            log.debug("KrakenExecutionHandler instantiated.")

        elif run_mode == "paper" or run_mode == "paper_trading":
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
            self.execution_handler = SimulatedExecutionHandler(
                config_manager=self.config,
                pubsub_manager=self.pubsub,
                data_service=self.historical_data_service,  # type: ignore[arg-type]
                logger_service=self.logger_service,
            )
            log.debug("SimulatedExecutionHandler instantiated for paper mode.")

        else:
            # Get supported modes from mode manager if available
            if self.mode_manager:
                supported_modes = [mode.value for mode in self.mode_manager.get_supported_modes()]
            else:
                supported_modes = ["live_trading", "paper_trading"]
            raise UnsupportedModeError(mode=run_mode, supported_modes=supported_modes)

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
        if self.execution_handler is None:
            raise RuntimeError("Execution handler should not be None at this point")
        return self.execution_handler  # type: ignore[return-value]

    async def initialize(self, args: argparse.Namespace) -> None:  # Add return type
        """Load configuration, set up logging, and instantiate components."""
        log.info("Initializing GalFridayApp (Version: %s)...", __version__)
        self.args = args  # args should be guaranteed by main_async

        # --- 1. Configuration Loading ---
        self._load_configuration(args.config)
        # No assertion needed, _load_configuration raises SystemExit on failure

        # --- 2. Enterprise-Grade Logging Setup (Replaces TODO line 923) ---
        try:
            # Initialize enterprise logging setup
            self.logging_setup = LoggingSetup(self.config)  # type: ignore
            logger = self.logging_setup.setup_logging(
                log_level=args.log_level,
                log_file=args.log_file,
                enable_json=args.enable_json_logs,
                enable_console=not args.disable_console_logs
            )
            log.info("Enterprise logging configured successfully.")
        except Exception:
            log.exception("ERROR: Failed to configure enterprise logging")
            # Fallback to basic logging
            setup_logging(self.config, args.log_level)
            log.info("Fallback to basic logging configuration.")

        # --- 3. Operational Mode Manager Setup (Replaces TODO line 922) ---
        try:
            # Initialize mode manager
            self.mode_manager = OperationalModeManager(self.config)  # type: ignore
            
            # Select and validate operational mode
            self.current_operational_mode = await self.mode_manager.select_and_initialize_operational_mode(
                explicit_mode=args.mode
            )
            log.info(f"Operational mode initialized: {self.current_operational_mode.value}")
            
        except Exception as e:
            log.exception("ERROR: Failed to initialize operational mode")
            # Fallback to safe default
            self.current_operational_mode = OperationalMode.PAPER_TRADING
            log.warning(f"Using fallback operational mode: {self.current_operational_mode.value}")

        # --- 4. Executor Setup ---
        self._setup_executor()

        # --- 5. PubSub Manager Instantiation ---
        self._instantiate_pubsub() # PubSub should be early
        # No assertion needed, _instantiate_pubsub raises SystemExit on failure

        # --- 6. Database Connection Pool and Session Maker ---
        if DatabaseConnectionPool is not None and self.config is not None:
            # Create a basic logger instance for DatabaseConnectionPool if self.logger_service isn't fully ready
            # Or ensure LoggerService is instantiated in a basic mode first.
            # For this step, assuming a basic logger from python's logging can be passed or self.logger_service is basic.
            # If LoggerService needs full setup for other services to use its get_logger, this order is tricky.
            # Let's assume self.logger_service is not yet the full DB-logging instance.
            # We will use a temporary logger for db_pool.
            temp_db_logger = logging.getLogger("gal_friday.db_pool_init")

            self.db_connection_pool = DatabaseConnectionPool(
                config=self.config,
                logger=temp_db_logger, # type: ignore # Pass a basic logger
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

        # --- 7. LoggerService Full Instantiation (with DB capabilities) ---
        if LoggerService is None:
            self._raise_logger_service_instantiation_failed()

        try:
            # Now instantiate the full LoggerService, passing the session_maker
            self.logger_service = LoggerService(
                config_manager=self.config, # type: ignore
                pubsub_manager=self.pubsub, # type: ignore
                db_session_maker=self.session_maker, # Pass the session_maker
            )
            self.services.append(self.logger_service) # Add to services for start/stop
            log.info("LoggerService instantiated/configured with DB support.")
            # If setup_logging was called earlier with a basic config, the LoggerService
            # might reconfigure handlers, or setup_logging should be called *after* this.
            # For simplicity, assume LoggerService internal _setup_logging handles this.
        except Exception as e:
            log.exception("FATAL: Failed to instantiate full LoggerService")
            raise LoggerServiceInstantiationFailedExit from e

        # --- 8. MigrationManager Setup ---
        if MigrationManager is not None and self.logger_service is not None:
            self.migration_manager = MigrationManager(
                logger=self.logger_service, # Pass the full logger service
                project_root_path="/app", # Explicitly set project root
            )
            log.info("MigrationManager instantiated.")
            try:
                log.info("Running database migrations to head...")
                await asyncio.to_thread(self.migration_manager.upgrade_to_head)
                log.info("Database migrations completed.")
            except Exception:
                log.exception("Failed to run database migrations.")
                raise # Re-raise as this is critical for app consistency
        else:
            log.critical("MigrationManager or LoggerService missing, cannot run migrations.")
            raise DependencyMissingError("Application", "MigrationManager or LoggerService")

        # --- 9. Other Service Instantiation (Order Matters!) ---
        # Services are initialized in dependency order.
        # Now pass session_maker to services that need it.
        # Example: self.portfolio_manager = PortfolioManager(..., session_maker=self.session_maker)
        # For now, these init methods are called; they would need internal updates
        # to accept and use the session_maker in subsequent refactoring.
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
            self._config_manager_instance,
            "start_watching",
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
        if self._config_manager_instance and hasattr(
            self._config_manager_instance,
            "stop_watching",
        ):
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
    # Handle special validation and health check modes
    try:
        # Quick check for help/version requests
        if len(sys.argv) > 1 and (sys.argv[1] in ['--help', '-h', '--version', '-V']):
            cli_parser = EnhancedCLIParser()
            cli_parser.parse_args()
            return
        
        # Parse arguments for validation and health check modes
        cli_parser = EnhancedCLIParser()
        args = cli_parser.parse_args()
        
        # Handle configuration validation
        if args.validate_config:
            print(f"Validating configuration file: {args.config}")
            try:
                if not os.path.exists(args.config):
                    print(f"❌ Configuration file not found: {args.config}")
                    sys.exit(1)
                
                # Create temporary config manager for validation
                temp_config = ConfigManager(
                    config_path=args.config,
                    logger_service=logging.getLogger("config_validator")
                )
                
                if temp_config.is_valid():
                    print("✅ Configuration validation successful")
                    sys.exit(0)
                else:
                    print("❌ Configuration validation failed")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"❌ Configuration validation error: {e}")
                sys.exit(1)
        
        # Handle health check
        if args.health_check:
            print("🏥 Performing system health check...")
            try:
                # Basic health checks
                health_status = {
                    'config_file_exists': os.path.exists(args.config),
                    'log_directory_writable': True,
                    'python_version_compatible': sys.version_info >= (3, 8),
                }
                
                if args.log_file:
                    try:
                        log_dir = Path(args.log_file).parent
                        log_dir.mkdir(parents=True, exist_ok=True)
                        test_file = log_dir / 'health_check_test.tmp'
                        test_file.touch()
                        test_file.unlink()
                    except Exception:
                        health_status['log_directory_writable'] = False
                
                # Report health status
                all_healthy = all(health_status.values())
                for check, status in health_status.items():
                    icon = "✅" if status else "❌"
                    print(f"   {icon} {check.replace('_', ' ').title()}: {'OK' if status else 'FAIL'}")
                
                if all_healthy:
                    print("✅ System health check passed")
                    sys.exit(0)
                else:
                    print("❌ System health check failed")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"❌ Health check error: {e}")
                sys.exit(1)
        
        # Normal application startup
        print(f"🌟 Starting Gal Friday Trading System v{__version__}")
        
        # Use asyncio.run for proper async handling
        asyncio.run(main_async(args))
        
    except KeyboardInterrupt:
        print("\n⏹️  Received keyboard interrupt, shutting down gracefully...")
        sys.exit(0)
    except SystemExit:
        # Let SystemExit pass through (from argparse, etc.)
        raise
    except Exception as e:
        print(f"💥 Fatal startup error: {e}")
        logging.exception("Fatal startup error")
        sys.exit(1)


if __name__ == "__main__":
    main()
