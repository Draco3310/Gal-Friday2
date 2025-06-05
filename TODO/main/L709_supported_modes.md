# Task: Replace example supported modes with configuration‑driven values; validate against ConfigManager.

### 1. Context
- **File:** `gal_friday/main.py`
- **Line:** `709`
- **Keyword/Pattern:** `"Example"`
- **Current State:** The code contains hardcoded example supported modes instead of reading from configuration management system.

### 2. Problem Statement
Hardcoded example supported modes create inflexibility in the main application entry point, making it difficult to change operational modes without code modifications. This approach violates configuration management best practices and prevents runtime mode configuration, deployment flexibility, and environment-specific settings. The lack of validation against ConfigManager also creates potential runtime errors when invalid modes are specified.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Mode Configuration Schema:** Define supported modes in configuration with validation rules
2. **Implement Mode Validation:** Validate requested modes against ConfigManager specifications
3. **Add Dynamic Mode Loading:** Load supported modes from configuration at runtime
4. **Create Mode Registry:** Centralized registry for mode definitions and capabilities
5. **Add Environment-Specific Modes:** Support different modes per deployment environment
6. **Implement Mode Compatibility Checks:** Ensure mode combinations are valid

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Set, Optional, Any
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
import logging

class OperationalMode(str, Enum):
    """Standardized operational modes"""
    LIVE_TRADING = "live_trading"
    PAPER_TRADING = "paper_trading"
    BACKTESTING = "backtesting"
    DATA_COLLECTION = "data_collection"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MONITORING_ONLY = "monitoring_only"
    MAINTENANCE = "maintenance"

@dataclass
class ModeCapabilities:
    """Capabilities and requirements for each mode"""
    requires_market_data: bool
    requires_execution_handler: bool
    requires_portfolio_manager: bool
    requires_risk_manager: bool
    requires_prediction_service: bool
    supports_real_money: bool
    resource_intensity: str  # 'low', 'medium', 'high'
    description: str

class ModeConfiguration(BaseModel):
    """Configuration for operational modes"""
    
    supported_modes: List[OperationalMode] = Field(
        description="List of supported operational modes"
    )
    
    default_mode: OperationalMode = Field(
        default=OperationalMode.PAPER_TRADING,
        description="Default mode if none specified"
    )
    
    mode_capabilities: Dict[str, ModeCapabilities] = Field(
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
    
    @validator('supported_modes')
    def validate_supported_modes(cls, v):
        if not v:
            raise ValueError("At least one operational mode must be supported")
        
        # Ensure all modes are valid enum values
        valid_modes = set(OperationalMode)
        for mode in v:
            if mode not in valid_modes:
                raise ValueError(f"Invalid mode: {mode}")
        
        return v
    
    @validator('default_mode')
    def validate_default_mode(cls, v, values):
        if 'supported_modes' in values and v not in values['supported_modes']:
            raise ValueError("Default mode must be in supported modes list")
        return v

class ModeManager:
    """Enterprise-grade mode management system"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.mode_config = self._load_mode_configuration()
        self.current_modes: Set[OperationalMode] = set()
        
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
            )
        }
    
    def get_supported_modes(self) -> List[OperationalMode]:
        """
        Replace hardcoded example modes with configuration-driven values
        Replace: example_modes = ["live", "paper", "backtest"]
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
    
    def validate_mode_request(self, requested_modes: List[str]) -> List[OperationalMode]:
        """Validate requested modes against configuration and compatibility rules"""
        
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
        self._validate_mode_compatibility(validated_modes)
        
        return validated_modes
    
    def _validate_mode_compatibility(self, modes: List[OperationalMode]) -> None:
        """Check for incompatible mode combinations"""
        
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
        """Determine which services are required for the given modes"""
        
        requirements = {
            'market_data': False,
            'execution_handler': False,
            'portfolio_manager': False,
            'risk_manager': False,
            'prediction_service': False
        }
        
        for mode in modes:
            capabilities = self.mode_config.mode_capabilities.get(
                mode.value, 
                self.default_capabilities.get(mode)
            )
            
            if capabilities:
                requirements['market_data'] |= capabilities.requires_market_data
                requirements['execution_handler'] |= capabilities.requires_execution_handler
                requirements['portfolio_manager'] |= capabilities.requires_portfolio_manager
                requirements['risk_manager'] |= capabilities.requires_risk_manager
                requirements['prediction_service'] |= capabilities.requires_prediction_service
        
        return requirements
    
    def _load_mode_configuration(self) -> ModeConfiguration:
        """Load mode configuration from ConfigManager"""
        
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

# Main function usage
def initialize_application_modes(config_manager) -> List[OperationalMode]:
    """Initialize application with configuration-driven modes"""
    
    mode_manager = ModeManager(config_manager)
    
    # Get supported modes from configuration (replaces hardcoded examples)
    supported_modes = mode_manager.get_supported_modes()
    
    # Log available modes for transparency
    logger = logging.getLogger(__name__)
    logger.info(f"Application supports {len(supported_modes)} operational modes:")
    for mode in supported_modes:
        capabilities = mode_manager.default_capabilities.get(mode)
        if capabilities:
            logger.info(f"  - {mode.value}: {capabilities.description}")
    
    return supported_modes 