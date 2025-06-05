# Task: Implement operational mode selection logic (live trading vs. simulation vs. backtesting).

### 1. Context
- **File:** `main.py`
- **Line:** `922`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing operational mode selection logic to choose between live trading, simulation, and backtesting modes.

### 2. Problem Statement
Without proper operational mode selection logic, the system cannot dynamically switch between different trading environments based on configuration or runtime parameters. This prevents flexible deployment across development, testing, and production environments, and limits the ability to run different types of analysis and trading operations.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Mode Configuration Framework:** Comprehensive configuration system for different operational modes
2. **Build Mode Selection Engine:** Intelligent mode selection based on environment, configuration, and runtime parameters
3. **Implement Mode-Specific Initialization:** Different initialization patterns for each operational mode
4. **Add Mode Validation:** Comprehensive validation to ensure mode compatibility with current environment
5. **Create Mode Switching Capability:** Safe switching between modes during runtime when appropriate
6. **Build Mode Monitoring:** Real-time monitoring of operational mode status and performance

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import os

class OperationalMode(str, Enum):
    LIVE_TRADING = "live_trading"
    PAPER_TRADING = "paper_trading"
    SIMULATION = "simulation"
    BACKTESTING = "backtesting"
    RESEARCH = "research"
    DEVELOPMENT = "development"

class EnvironmentType(str, Enum):
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    LOCAL = "local"

@dataclass
class ModeConfiguration:
    """Configuration for a specific operational mode"""
    mode: OperationalMode
    environment: EnvironmentType
    exchange_config: Dict[str, Any]
    data_source_config: Dict[str, Any]
    execution_config: Dict[str, Any]
    risk_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    required_services: List[str]
    optional_services: List[str]
    performance_targets: Dict[str, float]

@dataclass
class ModeValidationResult:
    """Result of operational mode validation"""
    is_valid: bool
    mode: OperationalMode
    environment: EnvironmentType
    validation_checks: Dict[str, bool]
    warnings: List[str]
    errors: List[str]
    missing_requirements: List[str]

class OperationalModeManager:
    """Enterprise-grade operational mode selection and management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.current_mode = None
        self.current_environment = None
        self.mode_configurations = {}
        self._initialize_mode_configurations()
    
    async def select_and_initialize_operational_mode(self) -> ModeConfiguration:
        """
        Select and initialize the appropriate operational mode
        Replace TODO with comprehensive mode selection logic
        """
        
        try:
            self.logger.info("Starting operational mode selection and initialization")
            
            # Detect current environment
            environment = self._detect_environment()
            self.logger.info(f"Detected environment: {environment.value}")
            
            # Determine operational mode
            mode = await self._determine_operational_mode(environment)
            self.logger.info(f"Selected operational mode: {mode.value}")
            
            # Get mode configuration
            mode_config = self._get_mode_configuration(mode, environment)
            
            # Validate mode compatibility
            validation_result = await self._validate_mode_compatibility(mode_config)
            
            if not validation_result.is_valid:
                error_msg = f"Mode validation failed: {validation_result.errors}"
                self.logger.error(error_msg)
                raise ModeSelectionError(error_msg)
            
            # Log warnings if any
            for warning in validation_result.warnings:
                self.logger.warning(f"Mode validation warning: {warning}")
            
            # Initialize mode-specific services
            await self._initialize_mode_services(mode_config)
            
            # Set current mode
            self.current_mode = mode
            self.current_environment = environment
            
            self.logger.info(
                f"Operational mode initialization complete: {mode.value} in {environment.value} environment"
            )
            
            return mode_config
            
        except Exception as e:
            self.logger.error(f"Error in operational mode selection: {e}")
            raise ModeSelectionError(f"Mode selection failed: {e}")
    
    def _detect_environment(self) -> EnvironmentType:
        """Detect current environment based on various indicators"""
        
        # Check environment variable first
        env_var = os.getenv('GAL_FRIDAY_ENV', '').lower()
        if env_var:
            try:
                return EnvironmentType(env_var)
            except ValueError:
                self.logger.warning(f"Invalid environment variable value: {env_var}")
        
        # Check configuration file
        config_env = self.config.get('environment', '').lower()
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
    
    async def _determine_operational_mode(self, environment: EnvironmentType) -> OperationalMode:
        """Determine operational mode based on environment and configuration"""
        
        # Check explicit mode configuration
        explicit_mode = self.config.get('operational_mode', '').lower()
        if explicit_mode:
            try:
                mode = OperationalMode(explicit_mode)
                self.logger.debug(f"Using explicit mode configuration: {mode.value}")
                return mode
            except ValueError:
                self.logger.warning(f"Invalid explicit mode configuration: {explicit_mode}")
        
        # Check command line arguments
        mode_from_args = await self._get_mode_from_args()
        if mode_from_args:
            return mode_from_args
        
        # Environment-based defaults
        if environment == EnvironmentType.PRODUCTION:
            return OperationalMode.LIVE_TRADING
        elif environment == EnvironmentType.STAGING:
            return OperationalMode.PAPER_TRADING
        elif environment == EnvironmentType.DEVELOPMENT:
            return OperationalMode.SIMULATION
        else:  # LOCAL
            return OperationalMode.BACKTESTING
    
    def _get_mode_configuration(self, mode: OperationalMode, environment: EnvironmentType) -> ModeConfiguration:
        """Get configuration for specific mode and environment"""
        
        config_key = f"{mode.value}_{environment.value}"
        
        if config_key in self.mode_configurations:
            return self.mode_configurations[config_key]
        
        # Build configuration dynamically
        base_config = self.config.get('modes', {}).get(mode.value, {})
        env_config = self.config.get('environments', {}).get(environment.value, {})
        
        # Merge configurations with environment taking precedence
        merged_config = {**base_config, **env_config}
        
        return ModeConfiguration(
            mode=mode,
            environment=environment,
            exchange_config=merged_config.get('exchange', {}),
            data_source_config=merged_config.get('data_source', {}),
            execution_config=merged_config.get('execution', {}),
            risk_config=merged_config.get('risk', {}),
            monitoring_config=merged_config.get('monitoring', {}),
            required_services=merged_config.get('required_services', []),
            optional_services=merged_config.get('optional_services', []),
            performance_targets=merged_config.get('performance_targets', {})
        )
    
    async def _validate_mode_compatibility(self, mode_config: ModeConfiguration) -> ModeValidationResult:
        """Validate that the selected mode is compatible with current environment"""
        
        validation_checks = {}
        warnings = []
        errors = []
        missing_requirements = []
        
        # Check required services availability
        for service in mode_config.required_services:
            is_available = await self._check_service_availability(service)
            validation_checks[f"service_{service}"] = is_available
            if not is_available:
                missing_requirements.append(service)
                errors.append(f"Required service not available: {service}")
        
        # Check exchange connectivity for live trading
        if mode_config.mode == OperationalMode.LIVE_TRADING:
            exchange_check = await self._validate_exchange_connectivity(mode_config.exchange_config)
            validation_checks['exchange_connectivity'] = exchange_check
            if not exchange_check:
                errors.append("Exchange connectivity required for live trading")
        
        # Check data source availability
        data_source_check = await self._validate_data_source(mode_config.data_source_config)
        validation_checks['data_source'] = data_source_check
        if not data_source_check:
            if mode_config.mode in [OperationalMode.LIVE_TRADING, OperationalMode.PAPER_TRADING]:
                errors.append("Real-time data source required for trading modes")
            else:
                warnings.append("Data source validation failed, may impact performance")
        
        # Check risk management configuration
        risk_check = await self._validate_risk_configuration(mode_config.risk_config)
        validation_checks['risk_management'] = risk_check
        if not risk_check:
            if mode_config.mode in [OperationalMode.LIVE_TRADING, OperationalMode.PAPER_TRADING]:
                errors.append("Risk management configuration required for trading modes")
            else:
                warnings.append("Risk management configuration incomplete")
        
        # Check environment-specific requirements
        env_check = await self._validate_environment_requirements(mode_config.environment)
        validation_checks['environment'] = env_check
        if not env_check:
            warnings.append(f"Environment requirements not fully met for {mode_config.environment.value}")
        
        is_valid = len(errors) == 0
        
        return ModeValidationResult(
            is_valid=is_valid,
            mode=mode_config.mode,
            environment=mode_config.environment,
            validation_checks=validation_checks,
            warnings=warnings,
            errors=errors,
            missing_requirements=missing_requirements
        )
    
    async def _initialize_mode_services(self, mode_config: ModeConfiguration) -> None:
        """Initialize services specific to the operational mode"""
        
        self.logger.info(f"Initializing services for {mode_config.mode.value} mode")
        
        # Initialize required services
        for service in mode_config.required_services:
            await self._initialize_service(service, mode_config)
        
        # Initialize optional services
        for service in mode_config.optional_services:
            try:
                await self._initialize_service(service, mode_config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize optional service {service}: {e}")
        
        # Mode-specific initialization
        if mode_config.mode == OperationalMode.LIVE_TRADING:
            await self._initialize_live_trading_services(mode_config)
        elif mode_config.mode == OperationalMode.PAPER_TRADING:
            await self._initialize_paper_trading_services(mode_config)
        elif mode_config.mode == OperationalMode.SIMULATION:
            await self._initialize_simulation_services(mode_config)
        elif mode_config.mode == OperationalMode.BACKTESTING:
            await self._initialize_backtesting_services(mode_config)
        elif mode_config.mode == OperationalMode.RESEARCH:
            await self._initialize_research_services(mode_config)
    
    def _is_production_environment(self) -> bool:
        """Check if running in production environment"""
        
        production_indicators = [
            os.getenv('IS_PRODUCTION', '').lower() == 'true',
            os.path.exists('/etc/gal-friday/production.conf'),
            'production' in os.getcwd().lower(),
            os.getenv('DEPLOYMENT_ENV', '').lower() == 'production'
        ]
        
        return any(production_indicators)
    
    def _is_staging_environment(self) -> bool:
        """Check if running in staging environment"""
        
        staging_indicators = [
            os.getenv('IS_STAGING', '').lower() == 'true',
            os.path.exists('/etc/gal-friday/staging.conf'),
            'staging' in os.getcwd().lower(),
            os.getenv('DEPLOYMENT_ENV', '').lower() == 'staging'
        ]
        
        return any(staging_indicators)
    
    def _is_development_environment(self) -> bool:
        """Check if running in development environment"""
        
        development_indicators = [
            os.getenv('IS_DEVELOPMENT', '').lower() == 'true',
            os.path.exists('pyproject.toml'),  # Development workspace indicator
            '.git' in os.listdir('.'),  # Git repository
            os.getenv('DEPLOYMENT_ENV', '').lower() == 'development'
        ]
        
        return any(development_indicators)
    
    async def get_current_mode_info(self) -> Dict[str, Any]:
        """Get information about current operational mode"""
        
        return {
            'mode': self.current_mode.value if self.current_mode else None,
            'environment': self.current_environment.value if self.current_environment else None,
            'initialized': self.current_mode is not None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

class ModeSelectionError(Exception):
    """Exception raised for operational mode selection errors"""
    pass
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of service initialization failures; fallback mode selection when preferred mode unavailable; comprehensive error logging
- **Configuration:** Environment-specific configuration files; override mechanisms for development/testing; mode-specific service configurations
- **Testing:** Unit tests for mode selection logic; integration tests for service initialization; environment detection validation; mode switching scenarios
- **Dependencies:** Configuration management system; service initialization framework; environment detection utilities; logging infrastructure

### 4. Acceptance Criteria
- [ ] Automatic environment detection accurately identifies production, staging, development, and local environments
- [ ] Mode selection logic chooses appropriate operational mode based on environment and configuration
- [ ] Mode validation ensures all required services and configurations are available before initialization
- [ ] Service initialization properly configures mode-specific components and dependencies
- [ ] Configuration framework supports environment-specific overrides and mode-specific settings
- [ ] Error handling provides clear feedback when mode selection or initialization fails
- [ ] Performance monitoring tracks mode initialization time and service startup metrics
- [ ] Integration tests verify successful initialization across all supported modes and environments
- [ ] Configuration validation prevents invalid mode/environment combinations
- [ ] TODO placeholder is completely replaced with production-ready implementation 