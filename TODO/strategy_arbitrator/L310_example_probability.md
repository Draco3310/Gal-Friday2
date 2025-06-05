# Task: Replace example probability check with configurable validation logic

### 1. Context
- **File:** `gal_friday/strategy_arbitrator.py`
- **Line:** `310`
- **Keyword/Pattern:** `"Example"`
- **Current State:** Hardcoded example probability validation that should be replaced with configurable validation logic

### 2. Problem Statement
The strategy arbitrator uses hardcoded example probability checks that are inflexible and not suitable for production use. This prevents the system from adapting to different trading strategies, risk tolerances, and market conditions that require dynamic validation criteria.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Validation Framework:** Configurable system for probability validation rules
2. **Build Rule Engine:** Dynamic rule evaluation system with multiple validation criteria
3. **Implement Validation Strategies:** Multiple approaches for probability validation
4. **Add Context-Aware Validation:** Market condition and strategy-specific validation
5. **Create Validation Monitoring:** Track validation performance and rule effectiveness
6. **Build Rule Management:** Runtime rule updates and validation rule versioning

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import time
import json

class ValidationOperator(str, Enum):
    """Validation operators for probability checks"""
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    BETWEEN = "between"
    NOT_BETWEEN = "not_between"
    IN_LIST = "in"
    NOT_IN_LIST = "not_in"

class ValidationLevel(str, Enum):
    """Validation severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationRule:
    """Single validation rule configuration"""
    name: str
    field_path: str
    operator: ValidationOperator
    value: Union[float, List[float], Any]
    level: ValidationLevel = ValidationLevel.ERROR
    message: Optional[str] = None
    enabled: bool = True
    conditions: Optional[Dict[str, Any]] = None

@dataclass
class ValidationContext:
    """Context for validation execution"""
    symbol: str
    strategy_id: str
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Result of validation execution"""
    is_valid: bool
    rule_name: str
    level: ValidationLevel
    message: str
    field_path: str
    actual_value: Any
    expected_value: Any
    timestamp: float = field(default_factory=time.time)

class ProbabilityValidator(ABC):
    """Abstract base class for probability validators"""
    
    @abstractmethod
    async def validate(self, data: Dict[str, Any], rule: ValidationRule, context: ValidationContext) -> ValidationResult:
        """Validate probability data against rule"""
        pass

class BasicProbabilityValidator(ProbabilityValidator):
    """Basic probability validation using simple operators"""
    
    async def validate(self, data: Dict[str, Any], rule: ValidationRule, context: ValidationContext) -> ValidationResult:
        """Validate using basic operators"""
        
        # Extract value from nested path
        actual_value = self._get_nested_value(data, rule.field_path)
        
        if actual_value is None:
            return ValidationResult(
                is_valid=False,
                rule_name=rule.name,
                level=rule.level,
                message=f"Field {rule.field_path} not found",
                field_path=rule.field_path,
                actual_value=None,
                expected_value=rule.value
            )
        
        # Apply validation operator
        is_valid = self._apply_operator(actual_value, rule.operator, rule.value)
        
        message = rule.message or f"Validation {rule.operator.value} failed for {rule.field_path}"
        if is_valid:
            message = f"Validation passed for {rule.field_path}"
        
        return ValidationResult(
            is_valid=is_valid,
            rule_name=rule.name,
            level=rule.level,
            message=message,
            field_path=rule.field_path,
            actual_value=actual_value,
            expected_value=rule.value
        )
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Extract value from nested dictionary using dot notation"""
        
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _apply_operator(self, actual: Any, operator: ValidationOperator, expected: Any) -> bool:
        """Apply validation operator"""
        
        try:
            if operator == ValidationOperator.GREATER_THAN:
                return actual > expected
            elif operator == ValidationOperator.GREATER_EQUAL:
                return actual >= expected
            elif operator == ValidationOperator.LESS_THAN:
                return actual < expected
            elif operator == ValidationOperator.LESS_EQUAL:
                return actual <= expected
            elif operator == ValidationOperator.EQUAL:
                return actual == expected
            elif operator == ValidationOperator.NOT_EQUAL:
                return actual != expected
            elif operator == ValidationOperator.BETWEEN:
                return expected[0] <= actual <= expected[1]
            elif operator == ValidationOperator.NOT_BETWEEN:
                return not (expected[0] <= actual <= expected[1])
            elif operator == ValidationOperator.IN_LIST:
                return actual in expected
            elif operator == ValidationOperator.NOT_IN_LIST:
                return actual not in expected
            else:
                return False
        except Exception:
            return False

class StatisticalProbabilityValidator(ProbabilityValidator):
    """Statistical validation using historical data"""
    
    def __init__(self):
        self.historical_data: Dict[str, List[float]] = {}
        self.max_history_size = 1000
    
    async def validate(self, data: Dict[str, Any], rule: ValidationRule, context: ValidationContext) -> ValidationResult:
        """Validate using statistical methods"""
        
        actual_value = self._get_nested_value(data, rule.field_path)
        
        if actual_value is None:
            return ValidationResult(
                is_valid=False,
                rule_name=rule.name,
                level=rule.level,
                message=f"Field {rule.field_path} not found",
                field_path=rule.field_path,
                actual_value=None,
                expected_value=rule.value
            )
        
        # Update historical data
        key = f"{context.symbol}_{rule.field_path}"
        if key not in self.historical_data:
            self.historical_data[key] = []
        
        self.historical_data[key].append(float(actual_value))
        
        # Maintain history size
        if len(self.historical_data[key]) > self.max_history_size:
            self.historical_data[key] = self.historical_data[key][-self.max_history_size:]
        
        # Apply statistical validation
        is_valid = self._validate_statistically(actual_value, rule, self.historical_data[key])
        
        return ValidationResult(
            is_valid=is_valid,
            rule_name=rule.name,
            level=rule.level,
            message=f"Statistical validation {'passed' if is_valid else 'failed'} for {rule.field_path}",
            field_path=rule.field_path,
            actual_value=actual_value,
            expected_value=rule.value
        )
    
    def _validate_statistically(self, value: float, rule: ValidationRule, history: List[float]) -> bool:
        """Validate using statistical methods"""
        
        if len(history) < 10:  # Not enough history
            return True
        
        import numpy as np
        
        if rule.operator == ValidationOperator.BETWEEN:
            # Check if value is within percentile range
            lower_percentile, upper_percentile = rule.value
            lower_bound = np.percentile(history, lower_percentile)
            upper_bound = np.percentile(history, upper_percentile)
            return lower_bound <= value <= upper_bound
        
        elif rule.operator == ValidationOperator.GREATER_THAN:
            # Check if value is above historical percentile
            percentile = rule.value
            threshold = np.percentile(history, percentile)
            return value > threshold
        
        # Default to basic validation
        return True

class ConfigurableProbabilityValidator:
    """Enterprise-grade configurable probability validator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Validator registry
        self.validators: Dict[str, ProbabilityValidator] = {
            'basic': BasicProbabilityValidator(),
            'statistical': StatisticalProbabilityValidator()
        }
        
        # Validation rules
        self.validation_rules: List[ValidationRule] = []
        self.rule_groups: Dict[str, List[ValidationRule]] = {}
        
        # Statistics
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'rule_executions': {},
            'performance_metrics': {}
        }
        
        # Configuration
        if config_path:
            self.load_validation_config(config_path)
    
    def load_validation_config(self, config_path: str) -> None:
        """Load validation configuration from file"""
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Parse validation rules
            self.validation_rules = []
            for rule_data in config_data.get('rules', []):
                rule = ValidationRule(
                    name=rule_data['name'],
                    field_path=rule_data['field_path'],
                    operator=ValidationOperator(rule_data['operator']),
                    value=rule_data['value'],
                    level=ValidationLevel(rule_data.get('level', 'error')),
                    message=rule_data.get('message'),
                    enabled=rule_data.get('enabled', True),
                    conditions=rule_data.get('conditions')
                )
                self.validation_rules.append(rule)
            
            # Parse rule groups
            self.rule_groups = {}
            for group_name, rule_names in config_data.get('rule_groups', {}).items():
                group_rules = [rule for rule in self.validation_rules if rule.name in rule_names]
                self.rule_groups[group_name] = group_rules
            
            self.logger.info(f"Loaded {len(self.validation_rules)} validation rules")
            
        except Exception as e:
            self.logger.error(f"Error loading validation configuration: {e}")
            raise
    
    async def validate_probability(self, data: Dict[str, Any], context: ValidationContext, 
                                 rule_group: Optional[str] = None) -> List[ValidationResult]:
        """
        Validate probability data with configurable rules
        Replace example probability check with comprehensive validation
        """
        
        try:
            self.validation_stats['total_validations'] += 1
            
            # Determine which rules to apply
            if rule_group and rule_group in self.rule_groups:
                rules_to_apply = self.rule_groups[rule_group]
            else:
                rules_to_apply = [rule for rule in self.validation_rules if rule.enabled]
            
            validation_results = []
            
            for rule in rules_to_apply:
                try:
                    # Check rule conditions
                    if not self._check_rule_conditions(rule, context):
                        continue
                    
                    # Execute validation
                    validator_type = rule.conditions.get('validator_type', 'basic') if rule.conditions else 'basic'
                    validator = self.validators.get(validator_type, self.validators['basic'])
                    
                    result = await validator.validate(data, rule, context)
                    validation_results.append(result)
                    
                    # Update statistics
                    self.validation_stats['rule_executions'][rule.name] = \
                        self.validation_stats['rule_executions'].get(rule.name, 0) + 1
                    
                    # Log validation result
                    if not result.is_valid and result.level == ValidationLevel.ERROR:
                        self.logger.error(f"Validation failed: {result.message}")
                    elif not result.is_valid and result.level == ValidationLevel.WARNING:
                        self.logger.warning(f"Validation warning: {result.message}")
                    
                except Exception as e:
                    self.logger.error(f"Error executing validation rule {rule.name}: {e}")
            
            # Update success/failure statistics
            failed_results = [r for r in validation_results if not r.is_valid and r.level == ValidationLevel.ERROR]
            
            if failed_results:
                self.validation_stats['failed_validations'] += 1
            else:
                self.validation_stats['successful_validations'] += 1
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error during probability validation: {e}")
            self.validation_stats['failed_validations'] += 1
            raise
    
    def _check_rule_conditions(self, rule: ValidationRule, context: ValidationContext) -> bool:
        """Check if rule conditions are met"""
        
        if not rule.conditions:
            return True
        
        # Check symbol condition
        if 'symbols' in rule.conditions:
            if context.symbol not in rule.conditions['symbols']:
                return False
        
        # Check strategy condition
        if 'strategies' in rule.conditions:
            if context.strategy_id not in rule.conditions['strategies']:
                return False
        
        # Check market conditions
        if 'market_conditions' in rule.conditions:
            for condition_key, condition_value in rule.conditions['market_conditions'].items():
                if context.market_conditions.get(condition_key) != condition_value:
                    return False
        
        return True
    
    def add_validation_rule(self, rule: ValidationRule) -> None:
        """Add new validation rule at runtime"""
        
        self.validation_rules.append(rule)
        self.logger.info(f"Added validation rule: {rule.name}")
    
    def remove_validation_rule(self, rule_name: str) -> bool:
        """Remove validation rule by name"""
        
        for i, rule in enumerate(self.validation_rules):
            if rule.name == rule_name:
                del self.validation_rules[i]
                self.logger.info(f"Removed validation rule: {rule_name}")
                return True
        
        return False
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation performance statistics"""
        
        total = self.validation_stats['total_validations']
        success_rate = (self.validation_stats['successful_validations'] / total * 100) if total > 0 else 0
        
        return {
            **self.validation_stats,
            'success_rate_percent': round(success_rate, 2),
            'total_rules': len(self.validation_rules),
            'enabled_rules': len([r for r in self.validation_rules if r.enabled]),
            'rule_groups': list(self.rule_groups.keys())
        }
    
    def register_custom_validator(self, name: str, validator: ProbabilityValidator) -> None:
        """Register custom validator"""
        
        self.validators[name] = validator
        self.logger.info(f"Registered custom validator: {name}")

# Example validation configuration
EXAMPLE_VALIDATION_CONFIG = {
    "rules": [
        {
            "name": "buy_probability_threshold",
            "field_path": "buy_probability",
            "operator": "gte",
            "value": 0.6,
            "level": "error",
            "message": "Buy probability must be at least 0.6",
            "enabled": True,
            "conditions": {
                "validator_type": "basic",
                "strategies": ["momentum", "mean_reversion"]
            }
        },
        {
            "name": "confidence_minimum",
            "field_path": "confidence",
            "operator": "gte",
            "value": 0.5,
            "level": "warning",
            "message": "Low confidence prediction",
            "enabled": True
        },
        {
            "name": "probability_range_check",
            "field_path": "buy_probability",
            "operator": "between",
            "value": [0.0, 1.0],
            "level": "error",
            "message": "Probability must be between 0 and 1",
            "enabled": True
        }
    ],
    "rule_groups": {
        "basic_checks": ["probability_range_check", "confidence_minimum"],
        "strategy_checks": ["buy_probability_threshold"]
    }
} 