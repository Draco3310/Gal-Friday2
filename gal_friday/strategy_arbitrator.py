"""Strategy arbitration for trading signal generation.

This module contains the StrategyArbitrator, which consumes prediction events from models,
applies trading strategy logic, and produces proposed trade signals. The arbitrator
supports configurable threshold strategies with secondary confirmation rules.
"""

# Strategy Arbitrator Module

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
import json
import operator  # Added for condition dispatch
from pathlib import Path
import time
from typing import Any, ClassVar
import uuid
from uuid import UUID

import asyncio

# Event imports
from .core.events import (
    EventType,
    PredictionEvent,
    TradeOutcomeEvent,
    TradeSignalProposedEvent,
)

# Import FeatureRegistryClient
from .core.feature_registry_client import FeatureRegistryClient

# Import PubSubManager
from .core.pubsub import PubSubManager
from .interfaces.service_protocol import ServiceProtocol

# Import LoggerService
from .logger_service import LoggerService

# Import MarketPriceService
from .market_price_service import MarketPriceService

# Import Strategy Selection System
from .strategy_selection import (
    StrategySelectionSystem,
)

# === Enterprise-Grade Prediction Interpretation Framework ===


class PredictionType(str, Enum):
    """Types of predictions supported by the interpretation system."""

    PROBABILITY = "probability"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SIGNAL = "signal"
    CONFIDENCE = "confidence"
    ENSEMBLE = "ensemble"


class InterpretationStrategy(str, Enum):
    """Strategies for interpreting predictions."""

    THRESHOLD_BASED = "threshold_based"
    PERCENTILE_BASED = "percentile_based"
    RELATIVE_STRENGTH = "relative_strength"
    WEIGHTED_AVERAGE = "weighted_average"
    CUSTOM = "custom"


@dataclass
class PredictionField:
    """Configuration for a single prediction field."""

    name: str
    type: PredictionType
    interpretation_strategy: InterpretationStrategy
    parameters: dict[str, Any] = field(default_factory=dict[str, Any])
    required: bool = True
    validation_rules: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class PredictionInterpretationConfig:
    """Complete configuration for prediction interpretation."""

    version: str
    description: str
    fields: list[PredictionField]
    default_interpretation: InterpretationStrategy
    fallback_rules: dict[str, Any] = field(default_factory=dict[str, Any])
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


class PredictionInterpreter:
    """Base class for prediction interpreters.

    Provides common functionality and defines the interface that all prediction
    interpreters should implement.

    This class implements the PredictionInterpreterProtocol from the interfaces module.
    """

    async def interpret(self, prediction: dict[str, Any], config: PredictionField) -> Any:
        """Interpret a prediction value according to configuration."""
        raise NotImplementedError("Subclasses must implement interpret()")

    def validate(self, prediction: dict[str, Any], config: PredictionField) -> bool:
        """Validate prediction against configuration."""
        raise NotImplementedError("Subclasses must implement validate()")


class ThresholdBasedInterpreter(PredictionInterpreter):
    """Threshold-based prediction interpreter for enterprise production use."""

    async def interpret(self, prediction: dict[str, Any], config: PredictionField) -> Any:
        """Interpret prediction using threshold-based logic."""
        field_name = config.name
        if field_name not in prediction:
            raise ValueError(f"Required field {field_name} not found in prediction")

        value = float(prediction[field_name])
        parameters = config.parameters

        if config.type == PredictionType.PROBABILITY:
            buy_threshold = parameters.get("buy_threshold", 0.6)
            sell_threshold = parameters.get("sell_threshold", 0.4)

            if value >= buy_threshold:
                return "BUY"
            if value <= sell_threshold:
                return "SELL"
            return "HOLD"

        if config.type == PredictionType.SIGNAL:
            threshold = parameters.get("threshold", 0.0)
            return "BUY" if value > threshold else "SELL"

        if config.type == PredictionType.CONFIDENCE:
            min_confidence = parameters.get("min_confidence", 0.5)
            return value >= min_confidence

        return value

    def validate(self, prediction: dict[str, Any], config: PredictionField) -> bool:
        """Validate prediction value against configured rules."""
        field_name = config.name
        if config.required and field_name not in prediction:
            return False

        if field_name in prediction:
            # First, try to convert to float
            try:
                value = float(prediction[field_name])
            except (ValueError, TypeError):
                return False

            # Now perform validation on the successfully converted value
            validation_rules = config.validation_rules

            # Check value range
            if "min_value" in validation_rules and value < validation_rules["min_value"]:
                return False
            if "max_value" in validation_rules and value > validation_rules["max_value"]:
                return False

        return True


# === Enterprise-Grade Validation Framework ===


class ValidationOperator(str, Enum):
    """Validation operators for probability and prediction checks."""

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
    """Validation severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationRule:
    """Single validation rule configuration."""

    name: str
    field_path: str
    operator: ValidationOperator
    value: float | list[float] | Any
    level: ValidationLevel = ValidationLevel.ERROR
    message: str | None = None
    enabled: bool = True
    conditions: dict[str, Any] | None = None


@dataclass
class ValidationContext:
    """Context for validation execution."""

    symbol: str
    strategy_id: str
    market_conditions: dict[str, Any] = field(default_factory=dict[str, Any])
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class ValidationResult:
    """Result of validation execution."""

    is_valid: bool
    rule_name: str
    level: ValidationLevel
    message: str
    field_path: str
    actual_value: Any
    expected_value: Any
    timestamp: float = field(default_factory=time.time)


class ProbabilityValidator:
    """Base class for probability validators."""

    async def validate(
        self,
        data: dict[str, Any],
        rule: ValidationRule,
        context: ValidationContext,
    ) -> ValidationResult:
        """Validate probability data against rule."""
        raise NotImplementedError("Subclasses must implement validate()")


class BasicProbabilityValidator(ProbabilityValidator):
    """Basic probability validation using configurable operators."""

    async def validate(
        self,
        data: dict[str, Any],
        rule: ValidationRule,
        context: ValidationContext,
    ) -> ValidationResult:
        """Validate using basic operators with enterprise error handling."""
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
                expected_value=rule.value,
            )

        # Apply validation operator
        is_valid = self._apply_operator(actual_value, rule.operator, rule.value)

        message = (
            rule.message
            or f"Validation {rule.operator.value} {'passed' if is_valid else 'failed'} for {rule.field_path}"
        )

        return ValidationResult(
            is_valid=is_valid,
            rule_name=rule.name,
            level=rule.level,
            message=message,
            field_path=rule.field_path,
            actual_value=actual_value,
            expected_value=rule.value,
        )

    def _get_nested_value(self, data: dict[str, Any], path: str) -> Any:
        """Extract value from nested dictionary using dot notation."""
        keys = path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    # Dispatch table for operators - defined as class variable for efficiency
    OPERATOR_DISPATCH: ClassVar[dict[ValidationOperator, Callable[[Any, Any], bool]]] = {
        ValidationOperator.GREATER_THAN: lambda a, e: float(a) > float(e),
        ValidationOperator.GREATER_EQUAL: lambda a, e: float(a) >= float(e),
        ValidationOperator.LESS_THAN: lambda a, e: float(a) < float(e),
        ValidationOperator.LESS_EQUAL: lambda a, e: float(a) <= float(e),
        ValidationOperator.EQUAL: lambda a, e: float(a) == float(e),
        ValidationOperator.NOT_EQUAL: lambda a, e: float(a) != float(e),
        ValidationOperator.BETWEEN: lambda a, e: bool(e[0] <= float(a) <= e[1]),
        ValidationOperator.NOT_BETWEEN: lambda a, e: not (e[0] <= float(a) <= e[1]),
        ValidationOperator.IN_LIST: lambda a, e: a in e,
        ValidationOperator.NOT_IN_LIST: lambda a, e: a not in e,
    }

    def _apply_operator(self, actual: Any, operator: ValidationOperator, expected: Any) -> bool:
        """Apply validation operator with comprehensive error handling."""
        try:
            handler = self.OPERATOR_DISPATCH.get(operator)
            if handler is None:
                self._raise_value_error(f"Unknown validation operator: {operator}")
                return False  # This line won't be reached due to exception, but satisfies type checker
            return handler(actual, expected)
        except (ValueError, TypeError, IndexError):
            return False

    def _raise_value_error(self, error_msg: str) -> None:
        """Helper method to raise ValueError, abstracting the raise from try block."""
        raise ValueError(error_msg)


class ConfigurableProbabilityValidator:
    """Enterprise-grade configurable probability validator with comprehensive monitoring."""

    def __init__(self, config_path: str | None = None, logger_service: LoggerService | None = None) -> None:
        """Initialize the instance."""
        self.logger = logger_service

        # Validator registry
        self.validators: dict[str, ProbabilityValidator] = {"basic": BasicProbabilityValidator()}

        # Validation rules
        self.validation_rules: list[ValidationRule] = []
        self.rule_groups: dict[str, list[ValidationRule]] = {}

        # Performance statistics
        self.validation_stats: dict[str, Any] = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "rule_executions": {},
            "performance_metrics": {},
        }

        # Configuration
        if config_path:
            self.load_validation_config(config_path)
        else:
            self._load_default_config()

    def load_validation_config(self, config_path: str) -> None:
        """Load validation configuration from file with enterprise error handling."""
        try:
            with Path(config_path).open() as f:
                config_data = json.load(f)

            # Parse validation rules
            self.validation_rules = []
            for rule_data in config_data.get("rules", []):
                rule = ValidationRule(
                    name=rule_data["name"],
                    field_path=rule_data["field_path"],
                    operator=ValidationOperator(rule_data["operator"]),
                    value=rule_data["value"],
                    level=ValidationLevel(rule_data.get("level", "error")),
                    message=rule_data.get("message"),
                    enabled=rule_data.get("enabled", True),
                    conditions=rule_data.get("conditions"),
                )
                self.validation_rules.append(rule)

            # Parse rule groups
            self.rule_groups = {}
            for group_name, rule_names in config_data.get("rule_groups", {}).items():
                group_rules = [rule for rule in self.validation_rules if rule.name in rule_names]
                self.rule_groups[group_name] = group_rules

            if self.logger:
                self.logger.info(
                    f"Loaded {len(self.validation_rules)} validation rules from {config_path}",
                )

        except Exception:
            if self.logger:
                self.logger.exception(f"Error loading validation configuration from {config_path}: ")
            raise

    def _load_default_config(self) -> None:
        """Load default validation configuration for production use."""
        default_rules = [
            ValidationRule(
                name="probability_range_check",
                field_path="prediction_value",
                operator=ValidationOperator.BETWEEN,
                value=[0.0, 1.0],
                level=ValidationLevel.ERROR,
                message="Probability must be between 0 and 1",
                enabled=True,
            ),
            ValidationRule(
                name="confidence_minimum",
                field_path="confidence",
                operator=ValidationOperator.GREATER_EQUAL,
                value=0.5,
                level=ValidationLevel.WARNING,
                message="Low confidence prediction detected",
                enabled=True,
            ),
        ]

        self.validation_rules = default_rules
        self.rule_groups = {
            "basic_checks": [rule for rule in default_rules if rule.name in ["probability_range_check"]],
            "quality_checks": [rule for rule in default_rules if rule.name in ["confidence_minimum"]],
        }

    async def validate_prediction(
        self,
        data: dict[str, Any],
        context: ValidationContext,
        rule_group: str | None = None,
    ) -> list[ValidationResult]:
        """Validate prediction data with configurable rules.

        Replaces hardcoded example probability checks with enterprise validation.
        """
        try:
            self.validation_stats["total_validations"] += 1
            rules_to_apply = self._determine_rules_to_apply(rule_group)
            validation_results = await self._execute_validation_rules(rules_to_apply, data, context)
            self._update_validation_statistics(validation_results)
        except Exception:
            if self.logger:
                self.logger.exception("Error during prediction validation: ")
            self.validation_stats["failed_validations"] += 1
            return []
        else:
            return validation_results

    def _determine_rules_to_apply(self, rule_group: str | None) -> list[ValidationRule]:
        """Determine which validation rules to apply based on rule group."""
        if rule_group and rule_group in self.rule_groups:
            return self.rule_groups[rule_group]
        return [rule for rule in self.validation_rules if rule.enabled]

    async def _execute_validation_rules(
        self,
        rules_to_apply: list[ValidationRule],
        data: dict[str, Any],
        context: ValidationContext,
    ) -> list[ValidationResult]:
        """Execute validation rules and collect results."""
        validation_results = []

        for rule in rules_to_apply:
            result = await self._execute_single_validation_rule(rule, data, context)
            if result:
                validation_results.append(result)

        return validation_results

    async def _execute_single_validation_rule(
        self,
        rule: ValidationRule,
        data: dict[str, Any],
        context: ValidationContext,
    ) -> ValidationResult | None:
        """Execute a single validation rule."""
        try:
            # Check rule conditions
            if not self._check_rule_conditions(rule, context):
                return None

            # Execute validation
            validator = self._get_validator_for_rule(rule)
            result = await validator.validate(data, rule, context)

            # Update statistics and log
            self._update_rule_execution_stats(rule)
            self._log_validation_result(result)

        except Exception:
            if self.logger:
                self.logger.exception(f"Error executing validation rule {rule.name}: ")
            return None
        else:
            return result

    def _get_validator_for_rule(self, rule: ValidationRule) -> ProbabilityValidator:
        """Get the appropriate validator for a rule."""
        validator_type = rule.conditions.get("validator_type", "basic") if rule.conditions else "basic"
        return self.validators.get(validator_type, self.validators["basic"])

    def _update_rule_execution_stats(self, rule: ValidationRule) -> None:
        """Update execution statistics for a rule."""
        self.validation_stats["rule_executions"][rule.name] = (
            self.validation_stats["rule_executions"].get(rule.name, 0) + 1
        )

    def _log_validation_result(self, result: ValidationResult) -> None:
        """Log validation result based on its severity."""
        if not result.is_valid and self.logger:
            if result.level == ValidationLevel.ERROR:
                self.logger.error(f"Validation failed: {result.message}")
            elif result.level == ValidationLevel.WARNING:
                self.logger.warning(f"Validation warning: {result.message}")

    def _update_validation_statistics(self, validation_results: list[ValidationResult]) -> None:
        """Update success/failure statistics based on validation results."""
        failed_results = [r for r in validation_results if not r.is_valid and r.level == ValidationLevel.ERROR]

        if failed_results:
            self.validation_stats["failed_validations"] += 1
        else:
            self.validation_stats["successful_validations"] += 1

    def _check_rule_conditions(self, rule: ValidationRule, context: ValidationContext) -> bool:
        """Check if rule conditions are met for context-aware validation."""
        if not rule.conditions:
            return True

        # Check symbol condition
        if "symbols" in rule.conditions and context.symbol not in rule.conditions["symbols"]:
            return False

        # Check strategy condition
        if "strategies" in rule.conditions and context.strategy_id not in rule.conditions["strategies"]:
            return False

        # Check market conditions
        if "market_conditions" in rule.conditions:
            for condition_key, condition_value in rule.conditions["market_conditions"].items():
                if context.market_conditions.get(condition_key) != condition_value:
                    return False

        return True

    def get_validation_statistics(self) -> dict[str, Any]:
        """Get comprehensive validation performance statistics."""
        total = self.validation_stats["total_validations"]
        success_rate = (self.validation_stats["successful_validations"] / total * 100) if total > 0 else 0

        return {
            **self.validation_stats,
            "success_rate_percent": round(success_rate, 2),
            "total_rules": len(self.validation_rules),
            "enabled_rules": len([r for r in self.validation_rules if r.enabled]),
            "rule_groups": list[Any](self.rule_groups.keys()),
        }


class PredictionInterpretationEngine:
    """Enterprise-grade prediction interpretation engine with configurable rules."""

    def __init__(self, config_path: str | None = None, logger_service: LoggerService | None = None) -> None:
        """Initialize the instance."""
        self.logger = logger_service

        # Interpreter registry
        self.interpreters: dict[InterpretationStrategy, PredictionInterpreter] = {
            InterpretationStrategy.THRESHOLD_BASED: ThresholdBasedInterpreter(),
        }

        # Configuration
        self.interpretation_config: PredictionInterpretationConfig | None = None

        # Statistics
        self.interpretation_stats = {
            "total_interpretations": 0,
            "successful_interpretations": 0,
            "validation_failures": 0,
            "interpretation_errors": 0,
        }

        # Load configuration
        if config_path:
            self.load_configuration(config_path)
        else:
            self._load_default_config()

    def load_configuration(self, config_path: str) -> None:
        """Load prediction interpretation configuration from file with enterprise error handling."""
        try:
            with Path(config_path).open() as f:
                config_data = json.load(f)

            # Parse configuration
            fields = []
            for field_data in config_data.get("fields", []):
                field = PredictionField(
                    name=field_data["name"],
                    type=PredictionType(field_data["type"]),
                    interpretation_strategy=InterpretationStrategy(
                        field_data["interpretation_strategy"],
                    ),
                    parameters=field_data.get("parameters", {}),
                    required=field_data.get("required", True),
                    validation_rules=field_data.get("validation_rules", {}),
                )
                fields.append(field)

            self.interpretation_config = PredictionInterpretationConfig(
                version=config_data["version"],
                description=config_data["description"],
                fields=fields,
                default_interpretation=InterpretationStrategy(
                    config_data.get("default_interpretation", "threshold_based"),
                ),
                fallback_rules=config_data.get("fallback_rules", {}),
                metadata=config_data.get("metadata", {}),
            )

            if self.logger:
                self.logger.info(
                    f"Loaded prediction interpretation configuration: {self.interpretation_config.description}",
                )

        except Exception:
            if self.logger:
                self.logger.exception(f"Error loading interpretation configuration from {config_path}: ")
            raise

    def _load_default_config(self) -> None:
        """Load default interpretation configuration for production use."""
        default_fields = [
            PredictionField(
                name="prediction_value",
                type=PredictionType.PROBABILITY,
                interpretation_strategy=InterpretationStrategy.THRESHOLD_BASED,
                parameters={"buy_threshold": 0.6, "sell_threshold": 0.4},
                required=True,
                validation_rules={"min_value": 0.0, "max_value": 1.0, "type": float},
            ),
        ]

        self.interpretation_config = PredictionInterpretationConfig(
            version="1.0",
            description="Default enterprise prediction interpretation configuration",
            fields=default_fields,
            default_interpretation=InterpretationStrategy.THRESHOLD_BASED,
            fallback_rules={"prediction_value": {"type": "default_value", "value": 0.5}},
        )

    async def interpret_prediction(self, prediction: dict[str, Any]) -> dict[str, Any]:
        """Interpret prediction according to loaded configuration with enterprise error handling."""
        if not self.interpretation_config:
            raise ValueError("No interpretation configuration loaded")

        try:
            self.interpretation_stats["total_interpretations"] += 1

            interpreted_result = {}
            validation_results: dict[str, Any] = {}

            # Process each configured field
            for field_config in self.interpretation_config.fields:
                field_result = await self._process_field(field_config, prediction, validation_results)
                if field_result is not None:
                    interpreted_result[field_config.name] = field_result

            # Add metadata
            interpreted_result["_metadata"] = self._create_metadata(validation_results)
            self.interpretation_stats["successful_interpretations"] += 1

        except Exception:
            self._handle_interpretation_error()
            raise
        else:
            return interpreted_result

    async def _process_field(
        self,
        field_config: PredictionField,
        prediction: dict[str, Any],
        validation_results: dict[str, bool],
    ) -> Any:
        """Process a single field configuration."""
        try:
            interpreter = self.interpreters[field_config.interpretation_strategy]

            # Validate field
            is_valid = interpreter.validate(prediction, field_config)
            validation_results[field_config.name] = is_valid

            if not is_valid:
                return self._handle_validation_failure(field_config)

            # Interpret field if present
            if field_config.name in prediction:
                return await interpreter.interpret(prediction, field_config)

        except Exception:
            return self._handle_field_error(field_config, prediction)
        else:
            return None

    def _handle_validation_failure(self, field_config: PredictionField) -> None:
        """Handle validation failure for a field."""
        self.interpretation_stats["validation_failures"] += 1
        if field_config.required and self.logger:
            self.logger.error(
                f"Required field {field_config.name} failed validation",
            )

    def _handle_field_error(self, field_config: PredictionField, prediction: dict[str, Any]) -> Any:
        """Handle errors during field processing."""
        self.interpretation_stats["interpretation_errors"] += 1
        if self.logger:
            self.logger.exception(f"Error interpreting field {field_config.name}: ")

        # Apply fallback rules
        return self._apply_fallback_rules(field_config.name, prediction)

    def _create_metadata(self, validation_results: dict[str, bool]) -> dict[str, Any]:
        """Create metadata for the interpretation result."""
        return {
            "interpretation_version": (
                self.interpretation_config.version if self.interpretation_config is not None else "unknown"
            ),
            "validation_results": validation_results,
            "timestamp": time.time(),
        }

    def _handle_interpretation_error(self) -> None:
        """Handle errors during overall interpretation."""
        self.interpretation_stats["interpretation_errors"] += 1
        if self.logger:
            self.logger.exception("Error interpreting prediction: ")

    def _apply_fallback_rules(self, field_name: str, prediction: dict[str, Any]) -> Any:
        """Apply fallback rules when interpretation fails."""
        if not self.interpretation_config:
            return None

        fallback_rules = self.interpretation_config.fallback_rules

        if field_name in fallback_rules:
            rule = fallback_rules[field_name]
            rule_type = rule.get("type", "default_value")

            if rule_type == "default_value":
                return rule.get("value")
            if rule_type == "copy_field":
                source_field = rule.get("source_field")
                return prediction.get(source_field)

        return None


# --- StrategyArbitrator Class ---
class StrategyArbitrator(ServiceProtocol):
    """Consumes prediction events from models, applies trading strategy logic, and produces proposed trade signals.

    The arbitrator supports configurable threshold-based strategies with secondary
    confirmation rules. These rules can leverage features published by the
    `FeatureEngine`. The `StrategyArbitrator` uses an injected `FeatureRegistryClient`
    to validate that feature names specified in confirmation rules exist in the
    Feature Registry during its configuration validation phase.

    During live operation, it expects `PredictionEvent`s to carry the necessary
    `triggering_features` (as a `dict[str, float]`) within their
    `associated_features` payload, which are then used to evaluate the
    confirmation rules.
    """

    def __init__(
        self,
        config: dict[str, Any],
        pubsub_manager: PubSubManager,
        logger_service: LoggerService,
        market_price_service: MarketPriceService,
        feature_registry_client: FeatureRegistryClient,  # Added parameter
        risk_manager: Any = None,  # Added for strategy selection
        portfolio_manager: Any = None,  # Added for strategy selection
        monitoring_service: Any = None,  # Added for strategy selection
        database_manager: Any = None,  # Added for strategy selection
    ) -> None:
        """Initialize the StrategyArbitrator.

        Args:
            config (dict[str, Any]): Configuration settings. Expected structure:
                strategy_arbitrator:
                  strategies:
                    - id: "mvp_threshold_v1"
                      buy_threshold: 0.65
                      # ... other strategy params ...
                      confirmation_rules: # Rules use feature names from Feature Registry
                        - feature: "rsi_14_default"
                          condition: "lt"
                          threshold: 30
                  prediction_interpretation:
                    config_path: "path/to/interpretation_config.json"  # Optional
                  validation:
                    config_path: "path/to/validation_config.json"      # Optional
                  strategy_selection:  # New configuration section
                    enabled: true
                    selection_frequency_hours: 4
                    criteria_weights:
                      performance_score: 0.40
                      risk_alignment: 0.35
                      market_fit: 0.20
                      operational_efficiency: 0.05
            pubsub_manager (PubSubManager): For subscribing/publishing events.
            logger_service (LoggerService): The shared logger instance.
            market_price_service (MarketPriceService): Service to get market prices.
            feature_registry_client (FeatureRegistryClient): Instance of the feature registry client.
            risk_manager: Risk manager instance for strategy selection integration.
            portfolio_manager: Portfolio manager instance for strategy selection integration.
            monitoring_service: Monitoring service instance for strategy selection integration.
            database_manager: Database manager for strategy performance data access.
        """
        # Initialize core components
        self._initialize_core_components(
            config,
            pubsub_manager,
            logger_service,
            market_price_service,
            feature_registry_client,
        )

        # Initialize interpretation and validation engines
        self._initialize_interpretation_validation()

        # Initialize strategy selection system
        self._initialize_strategy_selection(risk_manager, portfolio_manager, monitoring_service, database_manager)

        # Load and select strategies
        self._load_and_select_strategy()

        # Extract strategy parameters
        self._extract_strategy_parameters()

        # Validate configuration
        self._validate_and_finalize()

    def _initialize_core_components(
        self,
        config: dict[str, Any],
        pubsub_manager: PubSubManager,
        logger_service: LoggerService,
        market_price_service: MarketPriceService,
        feature_registry_client: FeatureRegistryClient,
    ) -> None:
        """Initialize core components and configuration."""
        self._config = config.get("strategy_arbitrator", {})
        self.pubsub = pubsub_manager
        self.logger = logger_service
        self.market_price_service = market_price_service
        self.feature_registry_client = feature_registry_client
        self._is_running = False
        self._main_task = None
        self._source_module = self.__class__.__name__
        self._prediction_handler = self.handle_prediction_event

    def _initialize_interpretation_validation(self) -> None:
        """Initialize enterprise-grade interpretation and validation systems."""
        interpretation_config_path = self._config.get("prediction_interpretation", {}).get(
            "config_path",
        )
        validation_config_path = self._config.get("validation", {}).get("config_path")

        self.prediction_interpretation_engine = PredictionInterpretationEngine(
            config_path=interpretation_config_path,
            logger_service=self.logger,
        )

        self.probability_validator = ConfigurableProbabilityValidator(
            config_path=validation_config_path,
            logger_service=self.logger,
        )

    def _initialize_strategy_selection(
        self,
        risk_manager: Any,
        portfolio_manager: Any,
        monitoring_service: Any,
        database_manager: Any,
    ) -> None:
        """Initialize Strategy Selection System if enabled and dependencies provided."""
        self._strategy_selection_enabled = self._config.get("strategy_selection", {}).get(
            "enabled",
            False,
        ) and all([risk_manager, portfolio_manager, monitoring_service, database_manager])

        if self._strategy_selection_enabled:
            selection_config = self._config.get("strategy_selection", {})
            self.strategy_selection_system = StrategySelectionSystem(
                logger=self.logger,
                config=selection_config,
                risk_manager=risk_manager,
                portfolio_manager=portfolio_manager,
                monitoring_service=monitoring_service,
                database_manager=database_manager,
            )
            self.logger.info("Strategy Selection System initialized and enabled", source_module=self._source_module)
        else:
            self.strategy_selection_system = None  # type: ignore[assignment]
            if self._config.get("strategy_selection", {}).get("enabled", False):
                self.logger.warning(
                    "Strategy selection enabled but dependencies not provided - using static selection",
                    source_module=self._source_module,
                )

    def _load_and_select_strategy(self) -> None:
        """Load strategies and select the primary strategy."""
        self._strategies = self._config.get("strategies", [])
        if not self._strategies:
            err_msg = "No strategies configured for StrategyArbitrator."
            self.logger.error(err_msg, source_module=self._source_module)
            raise StrategyConfigurationError(err_msg)

        # Select the best strategy based on configuration or use intelligent selection
        self._primary_strategy_config = self._select_best_strategy(self._strategies)
        self._strategy_id = self._primary_strategy_config.get("id", "default_strategy")

    def _extract_strategy_parameters(self) -> None:
        """Extract strategy parameters from the selected strategy configuration."""
        # Basic thresholds
        self._buy_threshold = Decimal(str(self._primary_strategy_config["buy_threshold"]))
        self._sell_threshold = Decimal(str(self._primary_strategy_config["sell_threshold"]))
        self._entry_type = self._primary_strategy_config.get("entry_type", "MARKET").upper()

        # Stop loss and take profit
        sl_pct_conf = self._primary_strategy_config.get("sl_pct")
        tp_pct_conf = self._primary_strategy_config.get("tp_pct")
        self._sl_pct = Decimal(str(sl_pct_conf)) if sl_pct_conf is not None else None
        self._tp_pct = Decimal(str(tp_pct_conf)) if tp_pct_conf is not None else None

        # Additional parameters
        self._confirmation_rules = self._primary_strategy_config.get("confirmation_rules", [])
        self._limit_offset_pct = Decimal(str(self._primary_strategy_config.get("limit_offset_pct", "0.0001")))
        self._prediction_interpretation = self._primary_strategy_config.get("prediction_interpretation", "directional")

        # Risk ratio
        default_rr_ratio_str = self._primary_strategy_config.get("stop_loss_to_take_profit_ratio", "1.0")
        self._stop_loss_to_take_profit_ratio = Decimal(default_rr_ratio_str)

        # Price change thresholds
        self._price_change_buy_threshold_pct = Decimal(
            str(self._primary_strategy_config.get("price_change_buy_threshold_pct", "0.01")),
        )
        self._price_change_sell_threshold_pct = Decimal(
            str(self._primary_strategy_config.get("price_change_sell_threshold_pct", "-0.01")),
        )

    def _validate_and_finalize(self) -> None:
        """Validate configuration and finalize initialization."""
        try:
            self._validate_configuration()
        except KeyError as key_error:
            self.logger.exception("Missing required strategy parameter.", source_module=self._source_module)
            raise StrategyConfigurationError from key_error
        except (InvalidOperation, TypeError) as value_error:
            self.logger.exception(
                "Invalid parameter format in strategy configuration.",
                source_module=self._source_module,
            )
            raise StrategyConfigurationError from value_error

    async def initialize(self, *args: Any, **kwargs: Any) -> None:
        """Async initialization hook for compatibility with ServiceProtocol."""
        # No asynchronous setup required at this time
        self.logger.debug("StrategyArbitrator initialization complete.", source_module=self._source_module)

    def _validate_core_parameters(self) -> None:
        """Validate core strategy parameters like entry type, thresholds, SL/TP percentages."""
        if self._entry_type not in ["MARKET", "LIMIT"]:
            self.logger.error(
                "Invalid entry_type in strategy: %s. Must be 'MARKET' or 'LIMIT'.",
                self._entry_type,
                source_module=self._source_module,
            )
            raise StrategyConfigurationError
        if self._buy_threshold <= self._sell_threshold:
            self.logger.error(
                "Buy threshold (%s) must be greater than sell threshold (%s).",
                self._buy_threshold,
                self._sell_threshold,
                source_module=self._source_module,
            )
            raise StrategyConfigurationError
        if self._sl_pct is not None and (self._sl_pct <= 0 or self._sl_pct >= 1):
            self.logger.error(
                "Stop-loss percentage (%s) must be between 0 and 1 (exclusive).",
                self._sl_pct,
                source_module=self._source_module,
            )
            raise StrategyConfigurationError
        if self._tp_pct is not None and (self._tp_pct <= 0 or self._tp_pct >= 1):
            self.logger.error(
                "Take-profit percentage (%s) must be between 0 and 1 (exclusive).",
                self._tp_pct,
                source_module=self._source_module,
            )
            raise StrategyConfigurationError
        if self._limit_offset_pct < 0:
            self.logger.error(
                "Limit offset percentage (%s) cannot be negative.",
                self._limit_offset_pct,
                source_module=self._source_module,
            )
            raise StrategyConfigurationError

    def _validate_prediction_interpretation_config(self) -> None:
        """Validate prediction interpretation settings using enterprise-grade configurable system.

        Replaces hardcoded example interpretations with configurable interpretation framework.
        """
        self._validate_interpretation_engine_initialized()
        config = self._validate_interpretation_config_loaded()
        supported_interpretations = self._get_supported_interpretations(config)
        self._validate_interpretation_is_supported(supported_interpretations)
        self._validate_field_specific_configuration(config)

        self.logger.info(
            "Prediction interpretation configuration validated successfully. "
            "Using enterprise framework with interpretation: %s",
            self._prediction_interpretation,
            source_module=self._source_module,
        )

    def _validate_interpretation_engine_initialized(self) -> None:
        """Validate that prediction interpretation engine is properly initialized."""
        if not self.prediction_interpretation_engine:
            self.logger.error("Prediction interpretation engine not initialized", source_module=self._source_module)
            raise StrategyConfigurationError("Prediction interpretation engine not initialized")

    def _validate_interpretation_config_loaded(self) -> PredictionInterpretationConfig:
        """Validate interpretation configuration is loaded and return it."""
        if not self.prediction_interpretation_engine.interpretation_config:
            self.logger.error("No prediction interpretation configuration loaded", source_module=self._source_module)
            raise StrategyConfigurationError("No prediction interpretation configuration loaded")
        return self.prediction_interpretation_engine.interpretation_config

    def _get_supported_interpretations(self, config: PredictionInterpretationConfig) -> list[str]:
        """Get list of supported interpretation types from configuration."""
        return [field.name for field in config.fields]

    def _validate_interpretation_is_supported(self, supported_interpretations: list[str]) -> None:
        """Validate current interpretation is supported."""
        if self._prediction_interpretation not in supported_interpretations:
            self._handle_unsupported_interpretation(supported_interpretations)

    def _handle_unsupported_interpretation(self, supported_interpretations: list[str]) -> None:
        """Handle unsupported interpretation with appropriate error messaging."""
        legacy_types = ["prob_up", "prob_down", "price_change_pct"]

        if self._prediction_interpretation in legacy_types:
            self.logger.error(
                "Legacy prediction_interpretation '%s' detected. "
                "Please update configuration to use enterprise interpretation framework. "
                "Supported interpretations: %s",
                self._prediction_interpretation,
                supported_interpretations,
                source_module=self._source_module,
            )
        else:
            self.logger.error(
                "Invalid prediction_interpretation in strategy: %s. Supported interpretations: %s",
                self._prediction_interpretation,
                supported_interpretations,
                source_module=self._source_module,
            )

        raise StrategyConfigurationError(
            f"Invalid prediction_interpretation: {self._prediction_interpretation}",
        )

    def _validate_field_specific_configuration(self, config: PredictionInterpretationConfig) -> None:
        """Validate field-specific configuration for the current interpretation."""
        for config_field in config.fields:
            if config_field.name == self._prediction_interpretation:
                self._validate_interpretation_field(config_field)
                break

    def _validate_interpretation_field(self, config_field: PredictionField) -> None:
        """Validate a specific interpretation field configuration."""
        if config_field.interpretation_strategy == InterpretationStrategy.THRESHOLD_BASED:
            self._validate_threshold_based_parameters(config_field)

    def _validate_threshold_based_parameters(self, config_field: PredictionField) -> None:
        """Validate parameters for threshold-based interpretation."""
        # Check required parameters
        required_params = ["buy_threshold", "sell_threshold"]
        missing_params = [p for p in required_params if p not in config_field.parameters]

        if missing_params:
            self.logger.error(
                "Missing required parameters for threshold-based interpretation: %s",
                missing_params,
                source_module=self._source_module,
            )
            raise StrategyConfigurationError(f"Missing parameters: {missing_params}")

        # Validate threshold consistency
        self._validate_threshold_consistency(config_field)

    def _validate_threshold_consistency(self, config_field: PredictionField) -> None:
        """Validate that buy threshold is greater than sell threshold."""
        if "buy_threshold" in config_field.parameters and "sell_threshold" in config_field.parameters:
            buy_thresh = config_field.parameters["buy_threshold"]
            sell_thresh = config_field.parameters["sell_threshold"]

            if buy_thresh <= sell_thresh:
                self.logger.error(
                    "Buy threshold (%s) must be greater than sell threshold (%s) for field %s",
                    buy_thresh,
                    sell_thresh,
                    config_field.name,
                    source_module=self._source_module,
                )
                raise StrategyConfigurationError("Invalid threshold configuration")

    def _validate_confirmation_rules_config(self) -> None:
        """Validate the structure of confirmation rules and check if feature names exist in the Feature Registry."""
        if not self.feature_registry_client or not self.feature_registry_client.is_loaded():
            self.logger.warning(
                "FeatureRegistryClient not available or not loaded. "
                "Skipping feature name validation in confirmation rules for strategy '%s'.",
                self._strategy_id,
                source_module=self._source_module,
            )
            # Perform only structural validation if registry is not available
            for rule in self._confirmation_rules:
                if not all(k in rule for k in ["feature", "condition", "threshold"]):
                    self.logger.error(
                        "Invalid confirmation rule structure for strategy '%s': %s",
                        self._strategy_id,
                        rule,
                        source_module=self._source_module,
                    )
                    raise StrategyConfigurationError(
                        f"Invalid confirmation rule structure for strategy {self._strategy_id}: {rule}",
                    )
            return

        for rule in self._confirmation_rules:
            if not all(k in rule for k in ["feature", "condition", "threshold"]):
                self.logger.error(
                    "Invalid confirmation rule structure for strategy '%s': %s",
                    self._strategy_id,
                    rule,
                    source_module=self._source_module,
                )
                raise StrategyConfigurationError(
                    f"Invalid confirmation rule structure for strategy {self._strategy_id}: {rule}",
                )

            feature_name = rule.get("feature")
            if feature_name:  # feature_name is present in the rule structure
                definition = self.feature_registry_client.get_feature_definition(str(feature_name))
                if definition is None:
                    self.logger.warning(
                        "Confirmation rule for strategy '%s' references feature_key '%s' "
                        "which is not found in the Feature Registry. This rule may be "
                        "ineffective or cause errors during secondary confirmation.",
                        self._strategy_id,
                        feature_name,
                        source_module=self._source_module,
                    )
                else:
                    self.logger.debug(
                        "Confirmation rule feature '%s' for strategy '%s' validated against Feature Registry.",
                        feature_name,
                        self._strategy_id,
                        source_module=self._source_module,
                    )
            # If feature_name is None or empty, the structural check above would have already caught it
            # if 'feature' was a required key with a non-empty value.
            # The current structural check `all(k in rule for k in ["feature", ...])` ensures 'feature' key exists.
            # An additional check for empty feature_name string might be useful if desired.

    def _validate_configuration(self) -> None:
        """Validate loaded strategy configuration by calling specific validators."""
        self._validate_core_parameters()
        self._validate_prediction_interpretation_config()
        self._validate_confirmation_rules_config()

        self.logger.info("Strategy configuration validated successfully.", source_module=self._source_module)

    async def _validate_prediction_event(self, event: PredictionEvent) -> bool:
        """Validate the incoming PredictionEvent using enterprise-grade configurable validation.

        Replaces hardcoded example probability checks with comprehensive validation framework.
        """
        # Basic structural validation
        if not hasattr(event, "prediction_value") or event.prediction_value is None:
            self.logger.warning(
                "PredictionEvent %s missing prediction_value.",
                event.event_id,
                source_module=self._source_module,
            )
            return False
        if not hasattr(event, "trading_pair") or not event.trading_pair:
            self.logger.warning(
                "PredictionEvent %s missing trading_pair.",
                event.event_id,
                source_module=self._source_module,
            )
            return False

        # Enterprise-grade validation using configurable rules
        try:
            # Create validation context
            validation_context = ValidationContext(
                symbol=event.trading_pair,
                strategy_id=self._strategy_id,
                market_conditions={},  # Can be extended with actual market data
                metadata={
                    "event_id": str(event.event_id),
                    "prediction_interpretation": self._prediction_interpretation,
                },
            )

            # Prepare data for validation
            prediction_data = {
                "prediction_value": event.prediction_value,
                "trading_pair": event.trading_pair,
            }

            # Add confidence if available
            if hasattr(event, "confidence") and event.confidence is not None:
                prediction_data["confidence"] = event.confidence

            # Run enterprise validation
            validation_results = await self.probability_validator.validate_prediction(
                data=prediction_data,
                context=validation_context,
                rule_group="basic_checks",  # Apply basic validation rules
            )

            # Check for validation failures
            critical_failures = [
                result for result in validation_results if not result.is_valid and result.level == ValidationLevel.ERROR
            ]

            if critical_failures:
                failure_messages = [result.message for result in critical_failures]
                self.logger.warning(
                    "PredictionEvent %s failed validation for %s: %s",
                    event.event_id,
                    event.trading_pair,
                    "; ".join(failure_messages),
                    source_module=self._source_module,
                )
                return False

            # Log warnings for non-critical failures
            warnings = [
                result
                for result in validation_results
                if not result.is_valid and result.level == ValidationLevel.WARNING
            ]

            for warning in warnings:
                self.logger.warning(
                    "PredictionEvent %s validation warning for %s: %s",
                    event.event_id,
                    event.trading_pair,
                    warning.message,
                    source_module=self._source_module,
                )

            # Additional business logic validation
            return self._validate_prediction_business_rules(event)

        except Exception:
            self.logger.exception(
                "Error during enterprise validation of PredictionEvent %s",
                event.event_id,
                source_module=self._source_module,
            )
            return False

    def _validate_prediction_business_rules(self, event: PredictionEvent) -> bool:
        """Additional business-specific validation rules for predictions."""
        try:
            # Validate prediction value can be converted to float
            val = float(event.prediction_value)

            # Business rule: Check for reasonable prediction values
            if val < -10 or val > 10:  # Reasonable bounds for most prediction types
                self.logger.warning(
                    "Prediction_value %s for %s appears unreasonable (outside [-10, 10]).",
                    val,
                    event.trading_pair,
                    source_module=self._source_module,
                )
                return False

        except ValueError:
            self.logger.warning(
                "Prediction_value %s is not a valid numeric value.",
                event.prediction_value,
                source_module=self._source_module,
            )
            return False
        else:
            return True

    def _calculate_stop_loss_price_and_risk(
        self,
        side: str,
        current_price: Decimal,
        # Optionally, tp_price can be provided to derive SL if sl_pct is not set
        tp_price_for_rr_calc: Decimal | None = None,
    ) -> tuple[Decimal | None, Decimal | None]:  # Returns (sl_price, risk_amount_per_unit)
        """Calculate stop-loss price and risk amount per unit."""
        sl_price: Decimal | None = None
        risk_amount_per_unit: Decimal | None = None

        if self._sl_pct is not None and self._sl_pct > 0:
            if side == "BUY":
                sl_price = current_price * (Decimal(1) - self._sl_pct)
                risk_amount_per_unit = current_price - sl_price
            elif side == "SELL":
                sl_price = current_price * (Decimal(1) + self._sl_pct)
                risk_amount_per_unit = sl_price - current_price
        elif (
            tp_price_for_rr_calc is not None  # Check if TP is provided for derivation
            and self._stop_loss_to_take_profit_ratio is not None
            and self._stop_loss_to_take_profit_ratio > 0
        ):
            if side == "BUY":
                reward_amount_per_unit = tp_price_for_rr_calc - current_price
                if reward_amount_per_unit > 0:  # Ensure positive reward
                    risk_amount_per_unit = reward_amount_per_unit / self._stop_loss_to_take_profit_ratio
                    sl_price = current_price - risk_amount_per_unit
            elif side == "SELL":
                reward_amount_per_unit = current_price - tp_price_for_rr_calc
                if reward_amount_per_unit > 0:  # Ensure positive reward
                    risk_amount_per_unit = reward_amount_per_unit / self._stop_loss_to_take_profit_ratio
                    sl_price = current_price + risk_amount_per_unit
        return sl_price, risk_amount_per_unit

    def _calculate_take_profit_price(
        self,
        side: str,
        current_price: Decimal,
        # Optionally, sl_price and risk can be provided to derive TP
        sl_price_for_rr_calc: Decimal | None = None,
        risk_amount_for_rr_calc: Decimal | None = None,
    ) -> Decimal | None:
        """Calculate take-profit price."""
        tp_price: Decimal | None = None

        if self._tp_pct is not None and self._tp_pct > 0:
            if side == "BUY":
                tp_price = current_price * (Decimal(1) + self._tp_pct)
            elif side == "SELL":
                tp_price = current_price * (Decimal(1) - self._tp_pct)
        elif (
            sl_price_for_rr_calc is not None
            and risk_amount_for_rr_calc is not None
            and risk_amount_for_rr_calc > 0
            and self._stop_loss_to_take_profit_ratio is not None
            and self._stop_loss_to_take_profit_ratio > 0
        ):
            reward_adjustment = risk_amount_for_rr_calc * self._stop_loss_to_take_profit_ratio
            if side == "BUY":
                tp_price = current_price + reward_adjustment
            elif side == "SELL":
                tp_price = current_price - reward_adjustment
        return tp_price

    async def _calculate_sl_tp_prices(
        self,
        side: str,
        current_price: Decimal,
        trading_pair: str,
    ) -> tuple[Decimal | None, Decimal | None]:
        """Calculate SL/TP prices based on configuration and current price."""
        if not self._validate_current_price(current_price, trading_pair):
            return None, None

        # Try to calculate SL/TP using two different approaches
        sl_price, tp_price = self._calculate_sl_tp_direct_approach(side, current_price)

        if not (sl_price and tp_price):
            sl_price, tp_price = self._calculate_sl_tp_derived_approach(side, current_price)

        if not self._validate_sl_tp_calculation_result(sl_price, tp_price, trading_pair):
            return None, None

        # Type narrowing: At this point, both sl_price and tp_price are validated to be not None
        if sl_price is None:
            return None, None
        if tp_price is None:
            return None, None

        if not self._validate_sl_tp_prices_logic(sl_price, tp_price, side, current_price, trading_pair):
            return None, None

        return self._finalize_sl_tp_prices(sl_price, tp_price, trading_pair)

    def _validate_current_price(self, current_price: Decimal, trading_pair: str) -> bool:
        """Validate that current price is valid for calculations."""
        if current_price <= 0:
            self.logger.error(
                "Cannot calculate SL/TP for %s: Invalid current_price %s",
                trading_pair,
                current_price,
                source_module=self._source_module,
            )
            return False
        return True

    def _calculate_sl_tp_direct_approach(
        self,
        side: str,
        current_price: Decimal,
    ) -> tuple[Decimal | None, Decimal | None]:
        """Calculate SL directly, then TP based on SL."""
        sl_price, risk_amount_per_unit = self._calculate_stop_loss_price_and_risk(side, current_price)
        tp_price = None

        if sl_price and risk_amount_per_unit:
            tp_price = self._calculate_take_profit_price(side, current_price, sl_price, risk_amount_per_unit)

        return sl_price, tp_price

    def _calculate_sl_tp_derived_approach(
        self,
        side: str,
        current_price: Decimal,
    ) -> tuple[Decimal | None, Decimal | None]:
        """Calculate TP directly, then derive SL from TP."""
        tp_price_direct = self._calculate_take_profit_price(side, current_price)

        if not tp_price_direct:
            return None, None

        # Try to derive SL using this TP
        derived_sl, derived_risk = self._calculate_stop_loss_price_and_risk(
            side,
            current_price,
            tp_price_for_rr_calc=tp_price_direct,
        )

        if derived_sl and derived_risk:
            return derived_sl, tp_price_direct

        return None, None

    def _validate_sl_tp_calculation_result(
        self,
        sl_price: Decimal | None,
        tp_price: Decimal | None,
        trading_pair: str,
    ) -> bool:
        """Validate that both SL and TP were successfully calculated."""
        if not sl_price:
            self.logger.error(
                "SL params error for %s on %s. Need sl_pct or (tp_pct & RR).",
                self._strategy_id,
                trading_pair,
                source_module=self._source_module,
            )

        if not tp_price:
            self.logger.error(
                "TP params error for %s on %s. Need tp_pct or (sl_pct & RR).",
                self._strategy_id,
                trading_pair,
                source_module=self._source_module,
            )

        if not (sl_price and tp_price):
            self.logger.error(
                "Failed to calculate both SL and TP prices for %s.",
                trading_pair,
                source_module=self._source_module,
            )
            return False

        return True

    def _validate_sl_tp_prices_logic(
        self,
        sl_price: Decimal,
        tp_price: Decimal,
        side: str,
        current_price: Decimal,
        trading_pair: str,
    ) -> bool:
        """Validate that SL/TP prices make logical sense for the trade side."""
        if side == "BUY":
            valid = sl_price < current_price and tp_price > current_price
        else:  # SELL
            valid = sl_price > current_price and tp_price < current_price

        if not valid:
            self.logger.error(
                "Invalid SL/TP for %s on %s: SL=%s, TP=%s, Cur=%s",
                side,
                trading_pair,
                sl_price,
                tp_price,
                current_price,
                source_module=self._source_module,
            )

        return valid

    def _finalize_sl_tp_prices(
        self,
        sl_price: Decimal,
        tp_price: Decimal,
        trading_pair: str,
    ) -> tuple[Decimal | None, Decimal | None]:
        """Finalize and return SL/TP prices with error handling."""
        try:
            return sl_price, tp_price
        except Exception:  # Should be rare as calculations are done, but for safety
            self.logger.exception(
                "Error during final SL/TP return for %s",
                trading_pair,
                source_module=self._source_module,
            )
            return None, None

    async def _determine_entry_price(self, side: str, current_price: Decimal, trading_pair: str) -> Decimal | None:
        """Determine the proposed entry price based on order type."""
        if self._entry_type == "MARKET":
            return None  # No specific price for market orders
        if self._entry_type == "LIMIT":
            try:
                # Fetch current spread
                spread_data = await self.market_price_service.get_bid_ask_spread(trading_pair)
                if (
                    spread_data is None
                    # or spread_data.get("bid") is None # spread_data is now a tuple[Any, ...] or None
                    # or spread_data.get("ask") is None
                ):
                    self.logger.warning(
                        "Cannot determine limit price for %s: Bid/Ask unavailable. Falling back to current price.",
                        trading_pair,
                        source_module=self._source_module,
                    )
                    return current_price  # Fallback as per whiteboard suggestion

                # Removed commented-out lines for ERA001
                best_bid, best_ask = spread_data  # Unpack tuple[Any, ...]

                if side == "BUY":
                    # Place limit slightly below current ask or at ask
                    limit_price = best_ask * (Decimal(1) - self._limit_offset_pct)
                    # Ensure buy limit is not above current_price (or ask)
                    # significantly if offset is large
                    limit_price = min(limit_price, best_ask)
                elif side == "SELL":
                    # Place limit slightly above current bid or at bid
                    limit_price = best_bid * (Decimal(1) + self._limit_offset_pct)
                    # Ensure sell limit is not below current_price (or bid) significantly
                    limit_price = max(limit_price, best_bid)
                else:  # Should not happen
                    self.logger.error(
                        "Invalid side '%s' for limit price determination.",  # G004 fix
                        side,
                        source_module=self._source_module,
                    )
                    return None

                # No rounding here, handled by downstream modules
            except Exception:
                self.logger.exception(
                    "Error determining limit price for %s. Falling back to current price.",
                    trading_pair,
                    source_module=self._source_module,
                )
                return current_price  # Fallback on error
            else:
                return limit_price  # TRY300 fix: moved from try block
        else:
            self.logger.error(
                "Unsupported entry type for price determination: %s",  # G004 fix
                self._entry_type,
                source_module=self._source_module,
            )
            return None

    _CONDITION_OPERATORS: ClassVar[dict[str, Callable[..., Any]]] = {
        "gt": operator.gt,
        "lt": operator.lt,
        "eq": operator.eq,
        "gte": operator.ge,
        "lte": operator.le,
        "ne": operator.ne,
    }

    def _validate_confirmation_rule(
        self,
        rule: dict[str, Any],
        features: dict[str, float],
        trading_pair: str,
        primary_side: str,
    ) -> bool:
        """Validate a single confirmation rule against the provided features.

        Args:
            rule: A dictionary defining the confirmation rule (feature, condition, threshold).
            features: A dictionary of feature names to their float values, typically
                      `triggering_features` from a `PredictionEvent`.
            trading_pair: The trading pair for which the rule is being validated.
            primary_side: The primary signal side ("BUY" or "SELL") being considered.

        Returns:
            True if the rule passes or is skipped (due to invalid structure or
            unsupported condition), False if the rule condition is not met or an
            error occurs during processing.
        """
        feature_name = rule.get("feature")
        condition_key = rule.get("condition")
        threshold_str = rule.get("threshold")

        if not all([feature_name, condition_key, threshold_str is not None]):  # threshold can be 0
            self.logger.warning(
                "Skipping invalid confirmation rule (missing component): %s for %s",
                rule,
                trading_pair,
                source_module=self._source_module,
            )
            return True  # Skip invalid rule, effectively passing it by not blocking

        # Optional: Validate feature_name against registry if desired for stricter checks
        # This would warn if features in rules are not registered in the feature registry

        feature_value_float = features.get(feature_name) if feature_name is not None else None

        if feature_value_float is None:
            self.logger.info(
                "Sec confirm failed for %s on %s: Feature '%s' not in triggering_features.",
                primary_side,
                trading_pair,
                feature_name,
                source_module=self._source_module,
            )
            return False

        rule_passes = False
        try:
            # Convert feature (float) and threshold (str) to Decimal for precise comparison
            feature_value_decimal = Decimal(str(feature_value_float))
            threshold = Decimal(str(threshold_str))

            op = self._CONDITION_OPERATORS.get(str(condition_key))

            if op:
                condition_met = op(feature_value_decimal, threshold)
                if condition_met:
                    rule_passes = True
                else:
                    self.logger.info(
                        "Secondary confirm failed for %s on %s: Rule %s %s %s (Val: %s) not met.",
                        primary_side,
                        trading_pair,
                        feature_name,
                        condition_key,  # Use renamed variable
                        threshold,
                        feature_value_decimal,
                        source_module=self._source_module,
                    )
                    rule_passes = False
            else:
                self.logger.warning(
                    "Unsupported condition '%s' in rule: %s for %s",
                    condition_key,  # Use renamed variable
                    rule,
                    trading_pair,
                    source_module=self._source_module,
                )
                rule_passes = True  # Skip rule with unsupported condition (treat as passed)

        except (InvalidOperation, TypeError, KeyError):
            self.logger.exception(
                "Error applying confirmation rule %s for %s",
                rule,
                trading_pair,
                source_module=self._source_module,
            )
            rule_passes = False  # Error in processing means rule fails

        return rule_passes

    def _apply_secondary_confirmation(self, prediction_event: PredictionEvent, primary_side: str) -> bool:
        """Check if secondary confirmation rules pass using features from the PredictionEvent.

        Args:
            prediction_event: The `PredictionEvent` containing associated features.
            primary_side: The primary signal side ("BUY" or "SELL") being considered.

        Returns:
            True if all confirmation rules pass or if no rules are defined.
            False if any rule fails or if necessary features are missing.

        Features are expected in `prediction_event.associated_features['triggering_features']`
        as a `dict[str, float]`.
        """
        if not self._confirmation_rules:
            return True  # No rules defined, confirmation passes by default

        associated_payload = getattr(prediction_event, "associated_features", None)
        raw_features: dict[str, float] | None = None
        if isinstance(associated_payload, dict):
            raw_features = associated_payload.get("triggering_features")

        if not raw_features or not isinstance(
            raw_features,
            dict,
        ):  # Check if raw_features is None, empty or not a dict
            self.logger.warning(
                "No valid 'triggering_features' (dict[str, float]) found in PredictionEvent %s for %s on %s.",
                prediction_event.event_id,
                primary_side,
                prediction_event.trading_pair,
                source_module=self._source_module,
            )
            return False  # Cannot confirm without features

        for rule in self._confirmation_rules:
            if not self._validate_confirmation_rule(
                rule,
                raw_features,  # Pass the dict[str, float]
                prediction_event.trading_pair,
                primary_side,
            ):
                return False  # Rule failed

        self.logger.debug(
            "All secondary confirmation rules passed for %s signal on %s.",
            primary_side,
            prediction_event.trading_pair,
            source_module=self._source_module,
        )
        return True

    def _get_side_from_prob_up(self, prob_up: Decimal) -> str | None:
        """Determine signal side based on probability of price increase."""
        if prob_up >= self._buy_threshold:
            return "BUY"
        if prob_up < self._sell_threshold:  # sell_threshold is upper bound for prob_up to sell
            return "SELL"
        return None

    def _get_side_from_prob_down(self, prob_down: Decimal, trading_pair: str) -> str | None:
        """Determine signal side based on probability of price decrease."""
        buy_signal: str | None = None
        # BUY condition based on effective P(up)
        effective_prob_up = Decimal(1) - prob_down
        if effective_prob_up >= self._buy_threshold:
            buy_signal = "BUY"

        sell_signal: str | None = None
        # SELL condition based on P(down) vs buy_threshold (Whiteboard rule)
        if prob_down >= self._buy_threshold:
            sell_signal = "SELL"

        if buy_signal and sell_signal:
            # Implies buy_thresh <= 0.5; rare for typical probability thresholds. (E501 fix)
            self.logger.warning(
                "Conflicting signals for %s using prob_down: BUY and SELL triggered. "
                "P(down)=%s, buy_threshold=%s. No signal generated.",
                trading_pair,
                prob_down,
                self._buy_threshold,
                source_module=self._source_module,
            )
            return None

        return buy_signal or sell_signal

    def _get_side_from_price_change_pct(self, price_change_pct: Decimal) -> str | None:
        """Determine signal side based on predicted price change percentage."""
        if price_change_pct >= self._price_change_buy_threshold_pct:
            return "BUY"
        if price_change_pct <= self._price_change_sell_threshold_pct:
            return "SELL"
        return None

    async def _calculate_signal_side(self, prediction_event: PredictionEvent) -> str | None:
        """Interpret prediction and determine the primary signal side using enterprise interpretation framework."""
        trading_pair = prediction_event.trading_pair

        try:
            # Build prediction data
            prediction_data = self._build_prediction_data(prediction_event)

            # Interpret prediction using configurable framework
            interpreted_result = await self.prediction_interpretation_engine.interpret_prediction(
                prediction_data,
            )

            # Extract and validate signal
            return self._extract_signal_from_interpretation(interpreted_result, trading_pair)

        except Exception as e:
            self._log_interpretation_failure(trading_pair, e)
            # Fallback to legacy interpretation logic for backward compatibility
            return self._legacy_calculate_signal_side(prediction_event)

    def _build_prediction_data(self, prediction_event: PredictionEvent) -> dict[str, Any]:
        """Build prediction data dictionary from prediction event."""
        prediction_data = {
            "prediction_value": prediction_event.prediction_value,
            "trading_pair": prediction_event.trading_pair,
        }

        # Add additional prediction data if available
        if hasattr(prediction_event, "confidence") and prediction_event.confidence is not None:
            prediction_data["confidence"] = prediction_event.confidence

        return prediction_data

    def _extract_signal_from_interpretation(self, interpreted_result: dict[str, Any], trading_pair: str) -> str | None:
        """Extract and validate signal from interpretation result."""
        if self._prediction_interpretation not in interpreted_result:
            self._log_missing_interpretation_field(trading_pair)
            return None

        signal = interpreted_result[self._prediction_interpretation]
        return self._validate_signal_format(signal, trading_pair)

    def _validate_signal_format(self, signal: Any, trading_pair: str) -> str | None:
        """Validate and convert signal to expected format."""
        # Handle string signals
        if isinstance(signal, str) and signal in ["BUY", "SELL", "HOLD"]:
            return signal if signal != "HOLD" else None

        # Handle boolean signals (confidence-type predictions)
        if isinstance(signal, bool):
            return "BUY" if signal else None

        # Invalid format
        self._log_unexpected_signal_format(signal, trading_pair)
        return None

    def _log_missing_interpretation_field(self, trading_pair: str) -> None:
        """Log when interpretation result is missing expected field."""
        self.logger.warning(
            "Interpretation result missing expected field '%s' for %s",
            self._prediction_interpretation,
            trading_pair,
            source_module=self._source_module,
        )

    def _log_unexpected_signal_format(self, signal: Any, trading_pair: str) -> None:
        """Log unexpected signal format."""
        self.logger.warning(
            "Unexpected signal format from interpretation engine: %s for %s",
            signal,
            trading_pair,
            source_module=self._source_module,
        )

    def _log_interpretation_failure(self, trading_pair: str, error: Exception) -> None:
        """Log interpretation failure."""
        self.logger.warning(
            "Enterprise interpretation failed for %s, falling back to legacy logic: %s",
            trading_pair,
            str(error),
            source_module=self._source_module,
        )

    def _legacy_calculate_signal_side(self, prediction_event: PredictionEvent) -> str | None:
        """Legacy signal calculation logic for backward compatibility."""
        trading_pair = prediction_event.trading_pair
        try:
            prediction_val = Decimal(str(prediction_event.prediction_value))
        except (InvalidOperation, TypeError):
            self.logger.warning(
                "Invalid prediction_value: '%s' for %s. Cannot determine side.",
                prediction_event.prediction_value,
                trading_pair,
                source_module=self._source_module,
            )
            return None

        interpretation = self._prediction_interpretation

        if interpretation == "prob_up":
            return self._get_side_from_prob_up(prediction_val)
        if interpretation == "prob_down":
            return self._get_side_from_prob_down(prediction_val, trading_pair)
        if interpretation == "price_change_pct":
            return self._get_side_from_price_change_pct(prediction_val)

        # Default for unknown interpretation
        self.logger.warning(
            "Unknown prediction_interpretation '%s'. Defaulting to 'prob_up' for %s.",
            interpretation,
            trading_pair,
            source_module=self._source_module,
        )
        return self._get_side_from_prob_up(prediction_val)

    async def _evaluate_strategy(self, prediction_event: PredictionEvent) -> TradeSignalProposedEvent | None:
        """Evaluate trading strategy based on prediction probabilities.

        Returns TradeSignalProposedEvent if strategy triggers, None otherwise.
        """
        try:
            trading_pair = prediction_event.trading_pair
            side = await self._calculate_signal_side(prediction_event)

            if not side:
                self._log_no_primary_signal(prediction_event)
                return None

            if not self._apply_secondary_confirmation(prediction_event, side):
                self._log_failed_secondary_confirmation(prediction_event, side)
                return None

            return await self._create_trade_signal_event(prediction_event, side, trading_pair)

        except (InvalidOperation, TypeError, AttributeError, Exception):
            self._log_strategy_evaluation_error(prediction_event)
            return None

    def _log_no_primary_signal(self, prediction_event: PredictionEvent) -> None:
        """Log when no primary signal is generated."""
        self.logger.debug(
            "No primary signal for %s (PredID: %s, Val: %s, Interpret: %s)",
            prediction_event.trading_pair,
            prediction_event.event_id,
            prediction_event.prediction_value,
            self._prediction_interpretation,
            source_module=self._source_module,
        )

    def _log_failed_secondary_confirmation(self, prediction_event: PredictionEvent, side: str) -> None:
        """Log when secondary confirmation fails."""
        self.logger.info(
            "Primary signal %s for %s (PredID: %s) failed secondary confirmation.",
            side,
            prediction_event.trading_pair,
            prediction_event.event_id,
            source_module=self._source_module,
        )

    async def _create_trade_signal_event(
        self,
        prediction_event: PredictionEvent,
        side: str,
        trading_pair: str,
    ) -> TradeSignalProposedEvent | None:
        """Create a trade signal event if all validations pass."""
        current_price = await self.market_price_service.get_latest_price(trading_pair)
        if current_price is None:
            self._log_price_fetch_failure(prediction_event)
            return None

        sl_price, tp_price = await self._calculate_sl_tp_prices(side, current_price, trading_pair)
        if sl_price is None or tp_price is None:
            self._log_sl_tp_calculation_failure(prediction_event, side, current_price)
            return None

        # Type narrowing: At this point, both sl_price and tp_price are not None
        return await self._build_trade_signal_event(
            prediction_event,
            side,
            trading_pair,
            current_price,
            sl_price,
            tp_price,
        )

    def _log_price_fetch_failure(self, prediction_event: PredictionEvent) -> None:
        """Log when current price cannot be fetched."""
        self.logger.warning(
            "Cannot generate signal for %s (PredID: %s): Failed to get current price.",
            prediction_event.trading_pair,
            prediction_event.event_id,
            source_module=self._source_module,
        )

    def _log_sl_tp_calculation_failure(
        self,
        prediction_event: PredictionEvent,
        side: str,
        current_price: Decimal,
    ) -> None:
        """Log when SL/TP calculation fails."""
        self.logger.warning(
            "Failed to calc SL/TP for %s on %s (PredID: %s). Price: %s",
            side,
            prediction_event.trading_pair,
            prediction_event.event_id,
            current_price,
            source_module=self._source_module,
        )

    async def _build_trade_signal_event(
        self,
        prediction_event: PredictionEvent,
        side: str,
        trading_pair: str,
        current_price: Decimal,
        sl_price: Decimal,
        tp_price: Decimal,
    ) -> TradeSignalProposedEvent:
        """Build the trade signal event with all required data."""
        proposed_entry = await self._determine_entry_price(side, current_price, trading_pair)
        signal_id = uuid.uuid4()

        event = TradeSignalProposedEvent(
            source_module=self._source_module,
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            signal_id=signal_id,
            trading_pair=trading_pair,
            exchange=prediction_event.exchange,
            side=side,
            entry_type=self._entry_type,
            proposed_sl_price=sl_price,
            proposed_tp_price=tp_price,
            strategy_id=self._strategy_id,
            proposed_entry_price=proposed_entry,
            triggering_prediction_event_id=prediction_event.event_id,
        )

        self._log_signal_generation(event, prediction_event.event_id, proposed_entry)
        return event

    def _log_signal_generation(
        self,
        event: TradeSignalProposedEvent,
        prediction_event_id: UUID,
        proposed_entry: Decimal | None,
    ) -> None:
        """Log successful signal generation."""
        self.logger.info(
            "Signal: %s (%s) for %s from PredID %s. SL:%s TP:%s Entry:%s",
            event.side,
            event.signal_id,
            event.trading_pair,
            prediction_event_id,
            event.proposed_sl_price,
            event.proposed_tp_price,
            proposed_entry or "MARKET",
            source_module=self._source_module,
        )

    def _log_strategy_evaluation_error(self, prediction_event: PredictionEvent) -> None:
        """Log errors during strategy evaluation."""
        event_id_str = getattr(prediction_event, "event_id", "UNKNOWN_EVENT_ID")
        pair_str = getattr(prediction_event, "trading_pair", "UNKNOWN_PAIR")
        self.logger.exception(
            "Error evaluating strategy for prediction %s on %s",
            event_id_str,
            pair_str,
            source_module=self._source_module,
        )

    async def start(self) -> None:
        """Start listening for prediction events and initialize strategy selection."""
        if self._is_running:
            self.logger.warning("StrategyArbitrator already running.", source_module=self._source_module)
            return
        self._is_running = True

        # Subscribe to PredictionEvent
        self.pubsub.subscribe(EventType.PREDICTION_GENERATED, self._prediction_handler)

        # Start strategy selection system if enabled
        if self._strategy_selection_enabled and self.strategy_selection_system:
            try:
                await self.strategy_selection_system.start()
                self.logger.info(
                    "Strategy Selection System started",
                    source_module=self._source_module,
                )
            except Exception:
                self.logger.exception("Failed to start Strategy Selection System: ", source_module=self._source_module)

        self.logger.info("StrategyArbitrator started.", source_module=self._source_module)

    async def stop(self) -> None:
        """Stop the event processing loop and strategy selection."""
        if not self._is_running:
            return
        self._is_running = False

        # Stop strategy selection system if running
        if self._strategy_selection_enabled and self.strategy_selection_system:
            try:
                await self.strategy_selection_system.stop()
                self.logger.info(
                    "Strategy Selection System stopped",
                    source_module=self._source_module,
                )
            except Exception:
                self.logger.exception("Error stopping Strategy Selection System: ", source_module=self._source_module)

        # Unsubscribe
        try:
            self.pubsub.unsubscribe(EventType.PREDICTION_GENERATED, self._prediction_handler)
            self.logger.info("Unsubscribed from PREDICTION_GENERATED.", source_module=self._source_module)
        except Exception:
            self.logger.exception("Error unsubscribing StrategyArbitrator", source_module=self._source_module)

        self.logger.info("StrategyArbitrator stopped.", source_module=self._source_module)

    async def handle_prediction_event(self, event: PredictionEvent) -> None:
        """Handle incoming prediction events directly."""
        # Type checking is handled by the type annotation
        # If we receive a non-PredictionEvent, it's a programming error

        # Validate prediction event using enterprise validation system
        if not await self._validate_prediction_event(event):
            # Validation failed, error logged in helper
            return

        if not self._is_running:
            self.logger.debug(
                "StrategyArbitrator is not running, skipping prediction event.",
                source_module=self._source_module,
            )
            return

        # Evaluate strategy based on the prediction event
        proposed_signal_event = await self._evaluate_strategy(event)

        # Publish the proposed signal event if generated
        if proposed_signal_event:
            await self._publish_trade_signal_proposed(proposed_signal_event)

    async def update_strategy_configuration(self, new_strategy_config: dict[str, Any]) -> bool:
        """Dynamically update strategy configuration during runtime.

        This method is called by the Strategy Selection System when transitioning
        to a new strategy. It updates all internal parameters without requiring
        a restart of the StrategyArbitrator.

        Args:
            new_strategy_config: The new strategy configuration dictionary

        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            self.logger.info(
                f"Updating strategy configuration from {self._strategy_id} to {new_strategy_config.get('id')}",
                source_module=self._source_module,
            )

            # Store previous configuration for potential rollback
            previous_config = {
                "strategy_id": self._strategy_id,
                "buy_threshold": self._buy_threshold,
                "sell_threshold": self._sell_threshold,
                "entry_type": self._entry_type,
                "sl_pct": self._sl_pct,
                "tp_pct": self._tp_pct,
                "confirmation_rules": self._confirmation_rules,
                "limit_offset_pct": self._limit_offset_pct,
                "prediction_interpretation": self._prediction_interpretation,
                "stop_loss_to_take_profit_ratio": self._stop_loss_to_take_profit_ratio,
                "price_change_buy_threshold_pct": self._price_change_buy_threshold_pct,
                "price_change_sell_threshold_pct": self._price_change_sell_threshold_pct,
            }

            # Update strategy parameters
            self._primary_strategy_config = new_strategy_config
            self._strategy_id = new_strategy_config.get("id", "default_strategy")

            # Update thresholds
            self._buy_threshold = Decimal(str(new_strategy_config["buy_threshold"]))
            self._sell_threshold = Decimal(str(new_strategy_config["sell_threshold"]))
            self._entry_type = new_strategy_config.get("entry_type", "MARKET").upper()

            # Update stop loss and take profit
            sl_pct_conf = new_strategy_config.get("sl_pct")
            tp_pct_conf = new_strategy_config.get("tp_pct")
            self._sl_pct = Decimal(str(sl_pct_conf)) if sl_pct_conf is not None else None
            self._tp_pct = Decimal(str(tp_pct_conf)) if tp_pct_conf is not None else None

            # Update other parameters
            self._confirmation_rules = new_strategy_config.get("confirmation_rules", [])
            self._limit_offset_pct = Decimal(str(new_strategy_config.get("limit_offset_pct", "0.0001")))
            self._prediction_interpretation = new_strategy_config.get("prediction_interpretation", "directional")
            default_rr_ratio_str = new_strategy_config.get("stop_loss_to_take_profit_ratio", "1.0")
            self._stop_loss_to_take_profit_ratio = Decimal(default_rr_ratio_str)

            # Update price change thresholds
            self._price_change_buy_threshold_pct = Decimal(
                str(new_strategy_config.get("price_change_buy_threshold_pct", "0.01")),
            )
            self._price_change_sell_threshold_pct = Decimal(
                str(new_strategy_config.get("price_change_sell_threshold_pct", "-0.01")),
            )

            # Validate new configuration
            try:
                self._validate_configuration()

                self.logger.info(
                    f"Successfully updated to strategy {self._strategy_id}",
                    source_module=self._source_module,
                )

                # Log key parameter changes
                self.logger.info(
                    f"New parameters - Buy: {self._buy_threshold}, Sell: {self._sell_threshold}, "
                    f"SL: {self._sl_pct}, TP: {self._tp_pct}, Entry: {self._entry_type}",
                    source_module=self._source_module,
                )

            except Exception:
                # Rollback on validation failure
                self.logger.exception(
                    "Strategy update validation failed. Rolling back.",
                    source_module=self._source_module,
                )

                # Restore previous configuration
                self._strategy_id = previous_config["strategy_id"]
                self._buy_threshold = previous_config["buy_threshold"]
                self._sell_threshold = previous_config["sell_threshold"]
                self._entry_type = previous_config["entry_type"]
                self._sl_pct = previous_config["sl_pct"]
                self._tp_pct = previous_config["tp_pct"]
                self._confirmation_rules = previous_config["confirmation_rules"]
                self._limit_offset_pct = previous_config["limit_offset_pct"]
                self._prediction_interpretation = previous_config["prediction_interpretation"]
                self._stop_loss_to_take_profit_ratio = previous_config["stop_loss_to_take_profit_ratio"]
                self._price_change_buy_threshold_pct = previous_config["price_change_buy_threshold_pct"]
                self._price_change_sell_threshold_pct = previous_config["price_change_sell_threshold_pct"]

                return False
            else:
                return True

        except Exception:
            self.logger.exception("Error updating strategy configuration: ", source_module=self._source_module)
            return False

    def get_current_strategy_info(self) -> dict[str, Any]:
        """Get information about the currently active strategy.

        Returns:
            Dictionary containing current strategy ID and key parameters
        """
        return {
            "strategy_id": self._strategy_id,
            "buy_threshold": float(self._buy_threshold),
            "sell_threshold": float(self._sell_threshold),
            "entry_type": self._entry_type,
            "sl_pct": float(self._sl_pct) if self._sl_pct else None,
            "tp_pct": float(self._tp_pct) if self._tp_pct else None,
            "has_confirmation_rules": bool(self._confirmation_rules),
            "num_confirmation_rules": len(self._confirmation_rules),
            "prediction_interpretation": self._prediction_interpretation,
        }

    async def report_trade_outcome(
        self,
        signal_id: str,
        outcome: str,
        pnl: Decimal | None = None,
        exit_reason: str | None = None,
    ) -> None:
        """Report trade outcome back to strategy selection system for learning.

        This allows the strategy selection system to track real-world performance
        of strategies and improve its selection decisions over time.

        Args:
            signal_id: The signal ID that resulted in the trade
            outcome: "win", "loss", or "breakeven"
            pnl: The profit/loss amount
            exit_reason: Why the trade was exited (e.g., "tp_hit", "sl_hit", "manual")
        """
        if not self._strategy_selection_enabled or not self.strategy_selection_system:
            return

        try:
            outcome_event = TradeOutcomeEvent.create(
                source_module=self._source_module,
                signal_id=uuid.UUID(signal_id),
                strategy_id=self._strategy_id,
                outcome=outcome,
                pnl=pnl,
                exit_reason=exit_reason,
            )

            await self.pubsub.publish(outcome_event)

            await self.logger.log_timeseries(
                measurement="trade_outcomes",
                tags={
                    "strategy_id": self._strategy_id,
                    "outcome": outcome,
                    "exit_reason": exit_reason or "unknown",
                },
                fields={"pnl": float(pnl) if pnl is not None else 0.0},
            )

            self.logger.info(
                f"Reported trade outcome for strategy {self._strategy_id}: "
                f"signal={signal_id}, outcome={outcome}, pnl={pnl}, exit={exit_reason}",
                source_module=self._source_module,
            )
        except Exception:
            self.logger.exception("Error reporting trade outcome: ", source_module=self._source_module)

    async def _publish_trade_signal_proposed(self, event: TradeSignalProposedEvent) -> None:
        """Publish the TradeSignalProposedEvent."""
        try:
            await self.pubsub.publish(event)
            self.logger.debug(
                "Published TradeSignalProposedEvent: %s for %s",
                event.signal_id,
                event.trading_pair,
                source_module=self._source_module,
            )
        except Exception:
            self.logger.exception(
                "Failed to publish TradeSignalProposedEvent %s for %s",
                event.signal_id,
                event.trading_pair,
                source_module=self._source_module,
            )

    def _select_best_strategy(self, strategies: list[dict[str, Any]]) -> dict[str, Any]:
        """Select the best strategy from available strategies using intelligent multi-criteria analysis.

        This method leverages the enterprise-grade StrategySelectionSystem when available,
        which evaluates strategies based on:
        - Historical performance metrics (Sharpe ratio, win rate, drawdown)
        - Current market conditions (volatility, trend, liquidity)
        - Risk alignment with portfolio constraints
        - Operational efficiency (latency, resource usage)

        Args:
            strategies: List of strategy configurations

        Returns:
            The selected strategy configuration
        """
        if not strategies:
            raise StrategyConfigurationError("No strategies available for selection")

        # Try intelligent selection first
        selected = self._try_intelligent_strategy_selection(strategies)
        if selected:
            return selected

        # Fallback to static selection
        return self._select_strategy_statically(strategies)

    def _try_intelligent_strategy_selection(self, strategies: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Try to select strategy using intelligent selection system."""
        if not (self._strategy_selection_enabled and self.strategy_selection_system):
            return None

        try:
            evaluation_result = self._evaluate_strategies_intelligently()
            if not evaluation_result:
                return None

            return self._process_intelligent_selection_result(evaluation_result, strategies)
        except Exception:
            self.logger.exception(
                "Error in intelligent strategy selection: . Falling back to static selection.",
                source_module=self._source_module,
            )
            return None

    def _evaluate_strategies_intelligently(self) -> Any:
        """Evaluate strategies using the intelligent system."""
        # Create a synchronous wrapper for the async evaluation
        loop = asyncio.get_event_loop()

        # Force immediate evaluation
        evaluation_result = loop.run_until_complete(
            self.strategy_selection_system.force_strategy_evaluation(),
        )

        if evaluation_result and evaluation_result.recommendation in ["deploy", "monitor"]:
            return evaluation_result
        return None

    def _process_intelligent_selection_result(
        self,
        evaluation_result: Any,
        strategies: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Process the result from intelligent strategy evaluation."""
        # Find the strategy configuration for the selected ID
        selected_config = next(
            (s for s in strategies if s.get("id") == evaluation_result.strategy_id),
            None,
        )

        if not selected_config:
            return None

        self._log_intelligent_selection_result(evaluation_result)
        return selected_config

    def _log_intelligent_selection_result(self, evaluation_result: Any) -> None:
        """Log the intelligent selection result and reasons."""
        self.logger.info(
            f"Intelligent strategy selection chose: {evaluation_result.strategy_id} "
            f"(score: {evaluation_result.composite_score:.3f}, "
            f"confidence: {evaluation_result.confidence_level:.2f})",
            source_module=self._source_module,
        )

        # Log selection reasons
        for reason in evaluation_result.reasons[:3]:  # Top 3 reasons
            self.logger.info(
                f"Selection reason: {reason}",
                source_module=self._source_module,
            )

    def _select_strategy_statically(self, strategies: list[dict[str, Any]]) -> dict[str, Any]:
        """Select strategy using static scoring when intelligent selection unavailable."""
        # Check if there's a preferred strategy marked in config
        preferred = self._find_preferred_strategy(strategies)
        if preferred:
            return preferred

        # Select based on static scoring
        return self._select_by_static_scoring(strategies)

    def _find_preferred_strategy(self, strategies: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Find a strategy marked as preferred in configuration."""
        preferred = next((s for s in strategies if s.get("preferred", False)), None)
        if preferred:
            self.logger.info(
                f"Selected preferred strategy: {preferred.get('id')}",
                source_module=self._source_module,
            )
        return preferred

    def _select_by_static_scoring(self, strategies: list[dict[str, Any]]) -> dict[str, Any]:
        """Select strategy based on static scoring of configuration parameters."""
        best_score = -1
        best_strategy = strategies[0]

        for strategy in strategies:
            score = self._calculate_static_strategy_score(strategy)
            if score > best_score:
                best_score = score
                best_strategy = strategy

        self.logger.info(
            f"Static strategy selection chose: {best_strategy.get('id')} (static score: {best_score})",
            source_module=self._source_module,
        )
        return best_strategy

    def _calculate_static_strategy_score(self, strategy: dict[str, Any]) -> int:
        """Calculate static score for a strategy based on configuration."""
        score = 0

        # Prefer strategies with both SL and TP defined
        if strategy.get("sl_pct") and strategy.get("tp_pct"):
            score += 2

        # Prefer strategies with reasonable thresholds
        buy_threshold = float(strategy.get("buy_threshold", 0.5))
        if 0.55 <= buy_threshold <= 0.70:
            score += 1

        # Prefer strategies with confirmation rules
        if strategy.get("confirmation_rules"):
            score += 1

        # Prefer limit orders over market for better execution
        if strategy.get("entry_type", "").upper() == "LIMIT":
            score += 1

        return score


# Custom Exception for configuration errors
class StrategyConfigurationError(ValueError):
    """Custom exception for strategy configuration errors."""
