# Task: Formalize prediction interpretations via configuration system

### 1. Context
- **File:** `gal_friday/strategy_arbitrator.py`
- **Line:** `191`
- **Keyword/Pattern:** `"Example"`
- **Current State:** Hardcoded example prediction interpretations that should be formalized through configuration

### 2. Problem Statement
The strategy arbitrator currently uses hardcoded example interpretations for predictions, making it inflexible and difficult to adapt to different trading strategies or prediction models. This limits the system's ability to handle diverse prediction formats and prevents runtime configuration of interpretation logic.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Prediction Interpretation Framework:** Configurable system for defining prediction formats
2. **Build Configuration Schema:** Structured configuration for prediction interpretation rules
3. **Implement Dynamic Interpretation:** Runtime interpretation based on configuration
4. **Add Validation System:** Validate prediction formats against configured schemas
5. **Create Interpretation Registry:** Plugin-based system for custom interpretation logic
6. **Build Testing Framework:** Comprehensive testing for interpretation accuracy

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from abc import ABC, abstractmethod

class PredictionType(str, Enum):
    """Types of predictions supported"""
    PROBABILITY = "probability"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SIGNAL = "signal"
    CONFIDENCE = "confidence"
    ENSEMBLE = "ensemble"

class InterpretationStrategy(str, Enum):
    """Strategies for interpreting predictions"""
    THRESHOLD_BASED = "threshold_based"
    PERCENTILE_BASED = "percentile_based"
    RELATIVE_STRENGTH = "relative_strength"
    WEIGHTED_AVERAGE = "weighted_average"
    CUSTOM = "custom"

@dataclass
class PredictionField:
    """Configuration for a single prediction field"""
    name: str
    type: PredictionType
    interpretation_strategy: InterpretationStrategy
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: bool = True
    validation_rules: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionInterpretationConfig:
    """Complete configuration for prediction interpretation"""
    version: str
    description: str
    fields: List[PredictionField]
    default_interpretation: InterpretationStrategy
    fallback_rules: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class PredictionInterpreter(ABC):
    """Abstract base class for prediction interpreters"""
    
    @abstractmethod
    async def interpret(self, prediction: Dict[str, Any], config: PredictionField) -> Any:
        """Interpret a prediction value according to configuration"""
        pass
    
    @abstractmethod
    def validate(self, prediction: Dict[str, Any], config: PredictionField) -> bool:
        """Validate prediction against configuration"""
        pass

class ThresholdBasedInterpreter(PredictionInterpreter):
    """Threshold-based prediction interpreter"""
    
    async def interpret(self, prediction: Dict[str, Any], config: PredictionField) -> Any:
        """Interpret prediction using threshold-based logic"""
        
        field_name = config.name
        if field_name not in prediction:
            raise ValueError(f"Required field {field_name} not found in prediction")
        
        value = prediction[field_name]
        parameters = config.parameters
        
        if config.type == PredictionType.PROBABILITY:
            buy_threshold = parameters.get('buy_threshold', 0.6)
            sell_threshold = parameters.get('sell_threshold', 0.4)
            
            if value >= buy_threshold:
                return 'BUY'
            elif value <= sell_threshold:
                return 'SELL'
            else:
                return 'HOLD'
        
        elif config.type == PredictionType.SIGNAL:
            threshold = parameters.get('threshold', 0.0)
            return 'BUY' if value > threshold else 'SELL'
        
        elif config.type == PredictionType.CONFIDENCE:
            min_confidence = parameters.get('min_confidence', 0.5)
            return value >= min_confidence
        
        return value
    
    def validate(self, prediction: Dict[str, Any], config: PredictionField) -> bool:
        """Validate prediction value"""
        
        field_name = config.name
        if config.required and field_name not in prediction:
            return False
        
        if field_name in prediction:
            value = prediction[field_name]
            validation_rules = config.validation_rules
            
            # Check value range
            if 'min_value' in validation_rules and value < validation_rules['min_value']:
                return False
            if 'max_value' in validation_rules and value > validation_rules['max_value']:
                return False
            
            # Check data type
            expected_type = validation_rules.get('type')
            if expected_type and not isinstance(value, expected_type):
                return False
        
        return True

class PercentileBasedInterpreter(PredictionInterpreter):
    """Percentile-based prediction interpreter"""
    
    def __init__(self):
        self.historical_values: Dict[str, List[float]] = {}
        self.max_history_size = 1000
    
    async def interpret(self, prediction: Dict[str, Any], config: PredictionField) -> Any:
        """Interpret prediction using percentile-based logic"""
        
        field_name = config.name
        value = prediction[field_name]
        parameters = config.parameters
        
        # Update historical values
        if field_name not in self.historical_values:
            self.historical_values[field_name] = []
        
        self.historical_values[field_name].append(float(value))
        
        # Maintain history size
        if len(self.historical_values[field_name]) > self.max_history_size:
            self.historical_values[field_name] = self.historical_values[field_name][-self.max_history_size:]
        
        # Calculate percentile
        import numpy as np
        
        history = self.historical_values[field_name]
        if len(history) < 10:  # Not enough history
            return 'HOLD'
        
        percentile = np.percentile(history, value * 100)
        
        buy_percentile = parameters.get('buy_percentile', 75)
        sell_percentile = parameters.get('sell_percentile', 25)
        
        current_percentile = (np.searchsorted(sorted(history), value) / len(history)) * 100
        
        if current_percentile >= buy_percentile:
            return 'BUY'
        elif current_percentile <= sell_percentile:
            return 'SELL'
        else:
            return 'HOLD'
    
    def validate(self, prediction: Dict[str, Any], config: PredictionField) -> bool:
        """Validate prediction value"""
        return ThresholdBasedInterpreter().validate(prediction, config)

class PredictionInterpretationEngine:
    """Enterprise-grade prediction interpretation engine with configurable rules"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Interpreter registry
        self.interpreters: Dict[InterpretationStrategy, PredictionInterpreter] = {
            InterpretationStrategy.THRESHOLD_BASED: ThresholdBasedInterpreter(),
            InterpretationStrategy.PERCENTILE_BASED: PercentileBasedInterpreter(),
        }
        
        # Configuration
        self.interpretation_config: Optional[PredictionInterpretationConfig] = None
        
        # Statistics
        self.interpretation_stats = {
            'total_interpretations': 0,
            'successful_interpretations': 0,
            'validation_failures': 0,
            'interpretation_errors': 0
        }
        
        # Load configuration
        if config_path:
            self.load_configuration(config_path)
    
    def load_configuration(self, config_path: str) -> None:
        """
        Load prediction interpretation configuration
        Replace example interpretations with formal configuration system
        """
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Parse configuration
            fields = []
            for field_data in config_data.get('fields', []):
                field = PredictionField(
                    name=field_data['name'],
                    type=PredictionType(field_data['type']),
                    interpretation_strategy=InterpretationStrategy(field_data['interpretation_strategy']),
                    parameters=field_data.get('parameters', {}),
                    required=field_data.get('required', True),
                    validation_rules=field_data.get('validation_rules', {})
                )
                fields.append(field)
            
            self.interpretation_config = PredictionInterpretationConfig(
                version=config_data['version'],
                description=config_data['description'],
                fields=fields,
                default_interpretation=InterpretationStrategy(config_data.get('default_interpretation', 'threshold_based')),
                fallback_rules=config_data.get('fallback_rules', {}),
                metadata=config_data.get('metadata', {})
            )
            
            self.logger.info(f"Loaded prediction interpretation configuration: {self.interpretation_config.description}")
            
        except Exception as e:
            self.logger.error(f"Error loading interpretation configuration: {e}")
            raise
    
    async def interpret_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret prediction according to loaded configuration"""
        
        if not self.interpretation_config:
            raise ValueError("No interpretation configuration loaded")
        
        try:
            self.interpretation_stats['total_interpretations'] += 1
            
            interpreted_result = {}
            validation_results = {}
            
            # Process each configured field
            for field_config in self.interpretation_config.fields:
                try:
                    # Validate field
                    interpreter = self.interpreters[field_config.interpretation_strategy]
                    is_valid = interpreter.validate(prediction, field_config)
                    validation_results[field_config.name] = is_valid
                    
                    if not is_valid:
                        self.interpretation_stats['validation_failures'] += 1
                        if field_config.required:
                            self.logger.error(f"Required field {field_config.name} failed validation")
                            continue
                    
                    # Interpret field
                    if field_config.name in prediction:
                        interpreted_value = await interpreter.interpret(prediction, field_config)
                        interpreted_result[field_config.name] = interpreted_value
                    
                except Exception as e:
                    self.interpretation_stats['interpretation_errors'] += 1
                    self.logger.error(f"Error interpreting field {field_config.name}: {e}")
                    
                    # Apply fallback rules
                    fallback_value = self._apply_fallback_rules(field_config.name, prediction)
                    if fallback_value is not None:
                        interpreted_result[field_config.name] = fallback_value
            
            # Add metadata
            interpreted_result['_metadata'] = {
                'interpretation_version': self.interpretation_config.version,
                'validation_results': validation_results,
                'timestamp': time.time()
            }
            
            self.interpretation_stats['successful_interpretations'] += 1
            return interpreted_result
            
        except Exception as e:
            self.interpretation_stats['interpretation_errors'] += 1
            self.logger.error(f"Error interpreting prediction: {e}")
            raise
    
    def _apply_fallback_rules(self, field_name: str, prediction: Dict[str, Any]) -> Any:
        """Apply fallback rules when interpretation fails"""
        
        fallback_rules = self.interpretation_config.fallback_rules
        
        if field_name in fallback_rules:
            rule = fallback_rules[field_name]
            rule_type = rule.get('type', 'default_value')
            
            if rule_type == 'default_value':
                return rule.get('value')
            elif rule_type == 'copy_field':
                source_field = rule.get('source_field')
                return prediction.get(source_field)
            elif rule_type == 'computed':
                # Apply computed fallback logic
                return self._compute_fallback_value(rule, prediction)
        
        return None
    
    def register_custom_interpreter(self, strategy: str, interpreter: PredictionInterpreter) -> None:
        """Register custom prediction interpreter"""
        
        self.interpreters[strategy] = interpreter
        self.logger.info(f"Registered custom interpreter: {strategy}")
    
    def get_interpretation_statistics(self) -> Dict[str, Any]:
        """Get interpretation performance statistics"""
        
        total = self.interpretation_stats['total_interpretations']
        success_rate = (self.interpretation_stats['successful_interpretations'] / total * 100) if total > 0 else 0
        
        return {
            **self.interpretation_stats,
            'success_rate_percent': round(success_rate, 2),
            'configuration_loaded': self.interpretation_config is not None,
            'registered_interpreters': list(self.interpreters.keys())
        }
    
    def validate_configuration(self, config_data: Dict[str, Any]) -> bool:
        """Validate interpretation configuration schema"""
        
        required_fields = ['version', 'description', 'fields']
        
        for field in required_fields:
            if field not in config_data:
                self.logger.error(f"Missing required field in configuration: {field}")
                return False
        
        # Validate fields structure
        for field_data in config_data.get('fields', []):
            if not all(key in field_data for key in ['name', 'type', 'interpretation_strategy']):
                self.logger.error(f"Invalid field configuration: {field_data}")
                return False
        
        return True

# Example configuration format
EXAMPLE_PREDICTION_CONFIG = {
    "version": "1.0",
    "description": "Default prediction interpretation configuration",
    "fields": [
        {
            "name": "buy_probability",
            "type": "probability",
            "interpretation_strategy": "threshold_based",
            "parameters": {
                "buy_threshold": 0.7,
                "sell_threshold": 0.3
            },
            "required": True,
            "validation_rules": {
                "min_value": 0.0,
                "max_value": 1.0,
                "type": float
            }
        },
        {
            "name": "confidence",
            "type": "confidence",
            "interpretation_strategy": "threshold_based",
            "parameters": {
                "min_confidence": 0.6
            },
            "required": False,
            "validation_rules": {
                "min_value": 0.0,
                "max_value": 1.0
            }
        }
    ],
    "default_interpretation": "threshold_based",
    "fallback_rules": {
        "buy_probability": {
            "type": "default_value",
            "value": 0.5
        }
    }
} 