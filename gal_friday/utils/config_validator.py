"""Configuration validation utilities for Gal-Friday.

This module provides comprehensive validation for configuration values to ensure
all settings are properly managed and no hardcoded values are used inappropriately.
Supports NFR-804 requirements for configuration management.

Enhanced with formal validation error framework and configuration guidance.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUGGESTION = "suggestion"


class ValidationCategory(str, Enum):
    """Categories of validation issues."""
    SYNTAX = "syntax"
    SCHEMA = "schema"
    DEPENDENCY = "dependency"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"
    BEST_PRACTICE = "best_practice"


@dataclass
class ValidationError:
    """Structured validation error with guidance."""
    code: str
    severity: ValidationSeverity
    category: ValidationCategory
    title: str
    message: str
    field_path: Optional[str] = None
    current_value: Optional[Any] = None
    expected_value: Optional[Any] = None
    remediation: Optional[str] = None
    documentation_url: Optional[str] = None
    examples: List[Dict[str, Any]] = field(default_factory=list[Any])
    related_errors: List[str] = field(default_factory=list[Any])


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool = False
    errors: List[ValidationError] = field(default_factory=list[Any])
    warnings: List[ValidationError] = field(default_factory=list[Any])
    suggestions: List[ValidationError] = field(default_factory=list[Any])
    validation_time: float = 0.0
    config_version: Optional[str] = None
    validator_version: str = "1.0.0"


class ConfigValidationErrorCodes:
    """Standard error codes for configuration validation."""
    
    # Critical errors
    MISSING_REQUIRED_FIELD = "CV001"
    INVALID_DATA_TYPE = "CV002"
    INVALID_ENUM_VALUE = "CV003"
    CIRCULAR_DEPENDENCY = "CV004"
    
    # Configuration errors
    INVALID_URL_FORMAT = "CV101"
    INVALID_PORT_NUMBER = "CV102"
    INVALID_FILE_PATH = "CV103"
    MISSING_DEPENDENCY = "CV104"
    CONFLICTING_SETTINGS = "CV105"
    
    # Security warnings
    WEAK_CREDENTIALS = "CV201"
    INSECURE_PROTOCOL = "CV202"
    EXPOSED_SECRETS = "CV203"
    PLACEHOLDER_CREDENTIALS = "CV204"
    
    # Performance warnings
    SUBOPTIMAL_SETTINGS = "CV301"
    RESOURCE_LIMITS = "CV302"
    
    # Best practice suggestions
    DEPRECATED_SETTING = "CV401"
    MISSING_OPTIONAL_FIELD = "CV402"
    INCONSISTENT_NAMING = "CV403"


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""


class ConfigurationGuidance:
    """Provides guidance for configuration validation errors."""
    
    def __init__(self) -> None:
        self.error_guidance = self._initialize_error_guidance()
        self.field_documentation = self._initialize_field_documentation()
    
    def _initialize_error_guidance(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error code guidance."""
        
        return {
            ConfigValidationErrorCodes.PLACEHOLDER_CREDENTIALS: {
                "title": "Placeholder Credentials Detected",
                "description": "Configuration contains placeholder credentials that must be replaced",
                "remediation_template": "Replace placeholder value for '{field}' with actual credentials or environment variable reference",
                "examples": [
                    {"field": "api_key", "placeholder": "YOUR_API_KEY", "correct": "${API_KEY}"},
                    {"field": "database_password", "placeholder": "YOUR_DB_PASSWORD", "correct": "${DB_PASSWORD}"}
                ],
                "documentation_url": "https://docs.gal-friday.com/config/credentials"
            },
            
            ConfigValidationErrorCodes.EXPOSED_SECRETS: {
                "title": "Hardcoded Secrets Detected",
                "description": "Sensitive values appear to be hardcoded instead of using environment variables",
                "remediation_template": "Move sensitive value '{field}' to environment variable ${env_var}",
                "examples": [
                    {"field": "api_secret", "wrong": "hardcoded-secret", "correct": "${API_SECRET}"},
                    {"field": "kraken.api_key", "wrong": "actual-key", "correct": "${KRAKEN_API_KEY}"}
                ],
                "documentation_url": "https://docs.gal-friday.com/security/environment-variables"
            },
            
            ConfigValidationErrorCodes.INVALID_URL_FORMAT: {
                "title": "Invalid URL Format",
                "description": "The URL format is incorrect or malformed",
                "remediation_template": "Correct the URL format for '{field}'. Expected format: {expected_format}",
                "examples": [
                    {"field": "kraken.api_url", "correct": "https://api.kraken.com", "incorrect": "api.kraken.com"},
                    {"field": "influx.url", "correct": "http://localhost:8086", "incorrect": "localhost:8086"}
                ],
                "documentation_url": "https://docs.gal-friday.com/config/url-format"
            },
            
            ConfigValidationErrorCodes.MISSING_REQUIRED_FIELD: {
                "title": "Missing Required Configuration Field",
                "description": "A required configuration field is missing",
                "remediation_template": "Add the required field '{field}' to your configuration",
                "examples": [
                    {"field": "exchange.kraken.api_url", "value": "https://api.kraken.com"},
                    {"field": "trading.pairs", "value": ["BTC/USD", "ETH/USD"]}
                ],
                "documentation_url": "https://docs.gal-friday.com/config/required-fields"
            }
        }
    
    def _initialize_field_documentation(self) -> Dict[str, Dict[str, Any]]:
        """Initialize field-specific documentation."""
        
        return {
            "kraken.api_key": {
                "description": "Kraken API key for trading operations",
                "type": "string",
                "sensitive": True,
                "env_var": "KRAKEN_API_KEY",
                "examples": ["${KRAKEN_API_KEY}"]
            },
            
            "kraken.api_secret": {
                "description": "Kraken API secret for authentication",
                "type": "string", 
                "sensitive": True,
                "env_var": "KRAKEN_API_SECRET",
                "examples": ["${KRAKEN_API_SECRET}"]
            },
            
            "database.password": {
                "description": "Database connection password",
                "type": "string",
                "sensitive": True,
                "env_var": "DB_PASSWORD", 
                "examples": ["${DB_PASSWORD}"]
            }
        }
    
    def get_error_guidance(self, error_code: str) -> Optional[Dict[str, Any]]:
        """Get guidance for specific error code."""
        return self.error_guidance.get(error_code)
    
    def get_field_documentation(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Get documentation for specific field."""
        return self.field_documentation.get(field_name)


class ConfigValidator:
    """Validates configuration values and ensures no inappropriate hardcoding."""

    # Known required configuration sections
    REQUIRED_SECTIONS: ClassVar[set[str]] = {
        "exchange",
        "trading",
        "portfolio",
        "risk",
        "monitoring",
        "logging",
        "prediction_service",
    }

    # URLs that should be configurable, not hardcoded
    CONFIGURABLE_URLS: ClassVar[dict[str, str]] = {
        "kraken.api_url": "https://api.kraken.com",
        "kraken.ws_url": "wss://ws.kraken.com/v2",
        "influx.url": "http://localhost:8086",
    }

    # Sensitive values that should come from environment
    SENSITIVE_KEYS: ClassVar[set[str]] = {
        "kraken.api_key",
        "kraken.api_secret",
        "database.password",
        "influx.token",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize validator with configuration data.

        Args:
            config: The configuration dictionary to validate
        """
        self.config = config
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.logger = logging.getLogger(__name__)
        self.guidance = ConfigurationGuidance()
        
        # Validation statistics
        self.validation_stats: Dict[str, Any] = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'error_counts': {},
            'most_common_errors': []
        }

    def validate_all(self) -> bool:
        """Run all validation checks.

        Returns:
            True if configuration is valid, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()

        self._validate_required_sections()
        self._validate_url_configuration()
        self._validate_sensitive_values()
        self._validate_risk_parameters()
        self._validate_trading_parameters()
        self._validate_logging_configuration()
        self._validate_prediction_service_config()

        return len(self.errors) == 0

    def validate_with_formal_errors(self) -> ValidationResult:
        """
        Enhanced validation with formal error reporting and guidance.
        Replaces placeholder warnings with structured validation.
        """
        
        start_time = time.time()
        
        try:
            self.validation_stats['total_validations'] += 1
            
            result = ValidationResult()
            
            # Perform various validation checks with structured errors
            self._validate_required_fields_enhanced(result)
            self._validate_url_configuration_enhanced(result)
            self._validate_sensitive_values_enhanced(result)
            self._validate_risk_parameters_enhanced(result)
            self._validate_trading_parameters_enhanced(result)
            self._validate_logging_configuration_enhanced(result)
            self._validate_prediction_service_config_enhanced(result)
            
            # Calculate validation time
            result.validation_time = time.time() - start_time
            
            # Determine overall validation status
            result.is_valid = len(result.errors) == 0
            
            # Update statistics
            if result.is_valid:
                self.validation_stats['successful_validations'] += 1
            else:
                self.validation_stats['failed_validations'] += 1
                
                # Track error frequencies
                for error in result.errors:
                    self.validation_stats['error_counts'][error.code] = \
                        self.validation_stats['error_counts'].get(error.code, 0) + 1
            
            self.logger.info(f"Configuration validation completed in {result.validation_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[self._create_critical_error("CV999", "Validation system error", str(e))]
            )

    def get_errors(self) -> list[str]:
        """Get all validation errors."""
        return self.errors.copy()

    def get_warnings(self) -> list[str]:
        """Get all validation warnings."""
        return self.warnings.copy()

    def _validate_required_sections(self) -> None:
        """Validate that all required configuration sections exist."""
        for section in self.REQUIRED_SECTIONS:
            if section not in self.config:
                self.errors.append(f"Missing required configuration section: {section}")
            elif not isinstance(self.config[section], dict):
                self.errors.append(f"Configuration section '{section}' must be a dictionary")

    def _validate_url_configuration(self) -> None:
        """Validate that URLs are properly configured, not hardcoded."""
        for key_path, default_url in self.CONFIGURABLE_URLS.items():
            sections = key_path.split(".")
            current = self.config

            # Navigate to the nested key
            try:
                for section in sections[:-1]:
                    current = current[section]
                final_key = sections[-1]

                if final_key not in current:
                    self.warnings.append(
                        f"URL not configured for '{key_path}', will use default: {default_url}")
                else:
                    configured_url = current[final_key]
                    if not isinstance(configured_url, str):
                        self.errors.append(
                            f"URL configuration '{key_path}' must be a string")
                    elif not self._is_valid_url(configured_url):
                        self.errors.append(
                            f"Invalid URL format for '{key_path}': {configured_url}")

            except KeyError:
                self.warnings.append(
                    f"Section missing for URL configuration '{key_path}', "
                    f"will use default: {default_url}")

    def _validate_sensitive_values(self) -> None:
        """Validate that sensitive values are properly handled."""
        for key_path in self.SENSITIVE_KEYS:
            sections = key_path.split(".")
            current = self.config

            # Check if sensitive value is directly in config (should be avoided)
            try:
                for section in sections[:-1]:
                    current = current[section]
                final_key = sections[-1]

                if final_key in current:
                    value = current[final_key]
                    if isinstance(value, str) and value.startswith("YOUR_"):
                        # REPLACED: Enhanced formal validation replaces placeholder warning
                        self.warnings.append(
                            f"[CV204] Placeholder credentials detected for '{key_path}'. "
                            f"Replace with environment variable reference like ${{{key_path.upper().replace('.', '_')}}}")
                    elif isinstance(value, str) and not value.startswith("${"):
                        # Check if it looks like a real secret vs environment variable reference
                        env_var = f"{key_path.upper().replace('.', '_')}"
                        if env_var not in os.environ:
                            self.warnings.append(
                                f"[CV203] Sensitive value '{key_path}' appears to be hardcoded. "
                                f"Move to environment variable ${env_var}")

            except KeyError:
                # Sensitive value not configured - this might be OK for some deployments
                pass

    def _validate_risk_parameters(self) -> None:
        """Validate risk management parameters."""
        if "risk" not in self.config:
            return

        risk_config = self.config["risk"]

        if "limits" in risk_config:
            limits = risk_config["limits"]

            # Validate percentage values
            percentage_keys = [
                "max_total_drawdown_pct",
                "max_daily_drawdown_pct",
                "max_weekly_drawdown_pct",
                "risk_per_trade_pct",
                "max_position_size_pct_equity",
                "max_total_exposure_pct_equity",
            ]

            for key in percentage_keys:
                if key in limits:
                    try:
                        max_risk_percent = 100
                        value = Decimal(str(limits[key]))
                        if value < 0:
                            self.errors.append(
                                f"Risk parameter '{key}' cannot be negative: {value}")
                        elif value > max_risk_percent:
                            self.warnings.append(
                                f"Risk parameter '{key}' is very high: {value}%")
                    except (ValueError, TypeError):
                        self.errors.append(f"Risk parameter '{key}' must be a valid number")

            # Validate consecutive losses
            if "max_consecutive_losses" in limits:
                try:
                    consecutive_losses_value = int(limits["max_consecutive_losses"])
                    if consecutive_losses_value < 1:
                        self.errors.append("max_consecutive_losses must be at least 1")
                except (ValueError, TypeError):
                    self.errors.append("max_consecutive_losses must be a valid integer")

    def _validate_trading_parameters(self) -> None:
        """Validate trading configuration parameters."""
        if "trading" not in self.config:
            return

        trading_config = self.config["trading"]

        # Validate trading pairs
        if "pairs" in trading_config:
            pairs = trading_config["pairs"]
            if not isinstance(pairs, list):
                self.errors.append("Trading pairs must be a list[Any]")
            elif len(pairs) == 0:
                self.errors.append("At least one trading pair must be configured")
            else:
                for pair in pairs:
                    if not isinstance(pair, str):
                        self.errors.append(
                            f"Trading pair must be a string: {pair}")
                    elif "/" not in pair:
                        self.errors.append(
                            f"Invalid trading pair format: {pair} "
                            "(expected format: BASE/QUOTE)")

    def _validate_logging_configuration(self) -> None:
        """Validate logging configuration."""
        if "logging" not in self.config:
            return

        logging_config = self.config["logging"]

        # Validate log level
        if "level" in logging_config:
            valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            level = logging_config["level"].upper()
            if level not in valid_levels:
                self.errors.append(f"Invalid log level: {level}. Must be one of {valid_levels}")

        # Validate database configuration if enabled
        if "database" in logging_config and logging_config["database"].get("enabled"):
            db_config = logging_config["database"]
            if "connection_string" in db_config:
                conn_str = db_config["connection_string"]
                if "YOUR_DB_PASSWORD" in conn_str:
                    self.warnings.append(
                        "Database connection string contains placeholder password. "
                        "Update with actual credentials or use environment variables.")

    def _validate_prediction_service_config(self) -> None:
        """Validate prediction service configuration."""
        if "prediction_service" not in self.config:
            return

        pred_config = self.config["prediction_service"]

        # Validate ensemble strategy
        if "ensemble_strategy" in pred_config:
            valid_strategies = {"none", "average", "weighted_average"}
            strategy = pred_config["ensemble_strategy"]
            if strategy not in valid_strategies:
                self.errors.append(
                    f"Invalid ensemble strategy: {strategy}. Must be one of {valid_strategies}")

        # Validate model configurations
        if "models" in pred_config:
            models = pred_config["models"]
            if not isinstance(models, list):
                self.errors.append("Prediction service models must be a list[Any]")
            else:
                for i, model in enumerate(models):
                    self._validate_model_config(model, i)

    def _validate_model_config(self, model: dict[str, Any], index: int) -> None:
        """Validate individual model configuration."""
        required_fields = ["model_id", "predictor_type", "model_path", "prediction_target"]

        for field in required_fields:
            if field not in model:
                self.errors.append(f"Model {index}: Missing required field '{field}'")

        # Validate predictor type
        if "predictor_type" in model:
            valid_types = {"xgboost", "sklearn", "lstm"}
            pred_type = model["predictor_type"]
            if pred_type not in valid_types:
                self.errors.append(
                    f"Model {index}: Invalid predictor type '{pred_type}'. "
                    f"Must be one of {valid_types}")

        # Validate LSTM-specific config
        if model.get("predictor_type") == "lstm":
            if "framework" not in model:
                self.errors.append(
                    f"Model {index}: LSTM models must specify 'framework' "
                    "(tensorflow or pytorch)")
            elif model["framework"] not in {"tensorflow", "pytorch"}:
                self.errors.append(
                    f"Model {index}: Invalid LSTM framework '{model['framework']}'. "
                    "Must be 'tensorflow' or 'pytorch'")

            if "sequence_length" not in model:
                self.errors.append(f"Model {index}: LSTM models must specify 'sequence_length'")
            elif not isinstance(model["sequence_length"], int) or model["sequence_length"] < 1:
                self.errors.append(f"Model {index}: sequence_length must be a positive integer")

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL format is valid."""
        url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
            r"localhost|"  # localhost...
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$", re.IGNORECASE)
        return bool(url_pattern.match(url))

    # Enhanced validation methods with formal error reporting
    
    def _validate_sensitive_values_enhanced(self, result: ValidationResult) -> None:
        """Enhanced validation of sensitive values with structured errors."""
        
        for key_path in self.SENSITIVE_KEYS:
            sections = key_path.split(".")
            current = self.config

            try:
                for section in sections[:-1]:
                    current = current[section]
                final_key = sections[-1]

                if final_key in current:
                    value = current[final_key]
                    if isinstance(value, str) and value.startswith("YOUR_"):
                        # Create structured error for placeholder credentials
                        error = self._create_validation_error(
                            code=ConfigValidationErrorCodes.PLACEHOLDER_CREDENTIALS,
                            severity=ValidationSeverity.WARNING,
                            category=ValidationCategory.SECURITY,
                            field_path=key_path,
                            current_value="[PLACEHOLDER]",
                            expected_value="environment variable reference"
                        )
                        result.warnings.append(error)
                        
                    elif isinstance(value, str) and not value.startswith("${"):
                        env_var = f"{key_path.upper().replace('.', '_')}"
                        if env_var not in os.environ:
                            # Create structured error for hardcoded secrets
                            error = self._create_validation_error(
                                code=ConfigValidationErrorCodes.EXPOSED_SECRETS,
                                severity=ValidationSeverity.WARNING,
                                category=ValidationCategory.SECURITY,
                                field_path=key_path,
                                current_value="[REDACTED]",
                                expected_value=f"${{{env_var}}}"
                            )
                            result.warnings.append(error)

            except KeyError:
                # Sensitive value not configured - this might be OK for some deployments
                pass

    def _validate_required_fields_enhanced(self, result: ValidationResult) -> None:
        """Enhanced validation of required fields with structured errors."""
        
        for section in self.REQUIRED_SECTIONS:
            if section not in self.config:
                error = self._create_validation_error(
                    code=ConfigValidationErrorCodes.MISSING_REQUIRED_FIELD,
                    severity=ValidationSeverity.CRITICAL,
                    category=ValidationCategory.SCHEMA,
                    field_path=section,
                    current_value=None,
                    expected_value="configuration section"
                )
                result.errors.append(error)
            elif not isinstance(self.config[section], dict):
                error = self._create_validation_error(
                    code=ConfigValidationErrorCodes.INVALID_DATA_TYPE,
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.SCHEMA,
                    field_path=section,
                    current_value=type(self.config[section]).__name__,
                    expected_value="dictionary"
                )
                result.errors.append(error)

    def _validate_url_configuration_enhanced(self, result: ValidationResult) -> None:
        """Enhanced URL validation with structured errors."""
        
        for key_path, default_url in self.CONFIGURABLE_URLS.items():
            sections = key_path.split(".")
            current = self.config

            try:
                for section in sections[:-1]:
                    current = current[section]
                final_key = sections[-1]

                if final_key not in current:
                    suggestion = ValidationError(
                        code=ConfigValidationErrorCodes.MISSING_OPTIONAL_FIELD,
                        severity=ValidationSeverity.SUGGESTION,
                        category=ValidationCategory.BEST_PRACTICE,
                        title="Optional URL Configuration Missing",
                        message=f"URL not configured for '{key_path}', will use default",
                        field_path=key_path,
                        current_value=None,
                        expected_value=default_url,
                        remediation=f"Add '{key_path}: {default_url}' to your configuration for explicit control"
                    )
                    result.suggestions.append(suggestion)
                else:
                    configured_url = current[final_key]
                    if not isinstance(configured_url, str):
                        error = self._create_validation_error(
                            code=ConfigValidationErrorCodes.INVALID_DATA_TYPE,
                            severity=ValidationSeverity.ERROR,
                            category=ValidationCategory.SCHEMA,
                            field_path=key_path,
                            current_value=type(configured_url).__name__,
                            expected_value="string"
                        )
                        result.errors.append(error)
                    elif not self._is_valid_url(configured_url):
                        error = self._create_validation_error(
                            code=ConfigValidationErrorCodes.INVALID_URL_FORMAT,
                            severity=ValidationSeverity.ERROR,
                            category=ValidationCategory.SYNTAX,
                            field_path=key_path,
                            current_value=configured_url,
                            expected_value="valid URL format"
                        )
                        result.errors.append(error)

            except KeyError:
                suggestion = ValidationError(
                    code=ConfigValidationErrorCodes.MISSING_OPTIONAL_FIELD,
                    severity=ValidationSeverity.SUGGESTION,
                    category=ValidationCategory.BEST_PRACTICE,
                    title="Configuration Section Missing for URL",
                    message=f"Section missing for URL configuration '{key_path}'",
                    field_path=key_path.split('.')[0],
                    current_value=None,
                    expected_value=f"section with {key_path}",
                    remediation=f"Add configuration section for {key_path} with value {default_url}"
                )
                result.suggestions.append(suggestion)

    def _validate_risk_parameters_enhanced(self, result: ValidationResult) -> None:
        """Enhanced risk parameter validation with structured errors."""
        
        if "risk" not in self.config:
            return

        risk_config = self.config["risk"]
        if "limits" in risk_config:
            limits = risk_config["limits"]
            percentage_keys = [
                "max_total_drawdown_pct",
                "max_daily_drawdown_pct", 
                "max_weekly_drawdown_pct",
                "risk_per_trade_pct",
                "max_position_size_pct_equity",
                "max_total_exposure_pct_equity",
            ]

            for key in percentage_keys:
                if key in limits:
                    try:
                        value = Decimal(str(limits[key]))
                        if value < 0:
                            error = self._create_validation_error(
                                code=ConfigValidationErrorCodes.INVALID_ENUM_VALUE,
                                severity=ValidationSeverity.ERROR,
                                category=ValidationCategory.SYNTAX,
                                field_path=f"risk.limits.{key}",
                                current_value=value,
                                expected_value="positive number"
                            )
                            result.errors.append(error)
                        elif value > 100:
                            warning = self._create_validation_error(
                                code=ConfigValidationErrorCodes.SUBOPTIMAL_SETTINGS,
                                severity=ValidationSeverity.WARNING,
                                category=ValidationCategory.PERFORMANCE,
                                field_path=f"risk.limits.{key}",
                                current_value=f"{value}%",
                                expected_value="< 100%"
                            )
                            result.warnings.append(warning)
                    except (ValueError, TypeError):
                        error = self._create_validation_error(
                            code=ConfigValidationErrorCodes.INVALID_DATA_TYPE,
                            severity=ValidationSeverity.ERROR,
                            category=ValidationCategory.SCHEMA,
                            field_path=f"risk.limits.{key}",
                            current_value=limits[key],
                            expected_value="numeric value"
                        )
                        result.errors.append(error)

    def _validate_trading_parameters_enhanced(self, result: ValidationResult) -> None:
        """Enhanced trading parameter validation with structured errors."""
        
        if "trading" not in self.config:
            return

        trading_config = self.config["trading"]
        if "pairs" in trading_config:
            pairs = trading_config["pairs"]
            if not isinstance(pairs, list):
                error = self._create_validation_error(
                    code=ConfigValidationErrorCodes.INVALID_DATA_TYPE,
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.SCHEMA,
                    field_path="trading.pairs",
                    current_value=type(pairs).__name__,
                    expected_value="list[Any]"
                )
                result.errors.append(error)
            elif len(pairs) == 0:
                error = self._create_validation_error(
                    code=ConfigValidationErrorCodes.MISSING_REQUIRED_FIELD,
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.SCHEMA,
                    field_path="trading.pairs",
                    current_value="empty list[Any]",
                    expected_value="at least one trading pair"
                )
                result.errors.append(error)

    def _validate_logging_configuration_enhanced(self, result: ValidationResult) -> None:
        """Enhanced logging configuration validation with structured errors."""
        
        if "logging" not in self.config:
            return

        logging_config = self.config["logging"]
        if "level" in logging_config:
            valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            level = logging_config["level"].upper()
            if level not in valid_levels:
                error = self._create_validation_error(
                    code=ConfigValidationErrorCodes.INVALID_ENUM_VALUE,
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.SYNTAX,
                    field_path="logging.level",
                    current_value=level,
                    expected_value=f"one of {valid_levels}"
                )
                result.errors.append(error)

    def _validate_prediction_service_config_enhanced(self, result: ValidationResult) -> None:
        """Enhanced prediction service validation with structured errors."""
        
        if "prediction_service" not in self.config:
            return

        pred_config = self.config["prediction_service"]
        if "ensemble_strategy" in pred_config:
            valid_strategies = {"none", "average", "weighted_average"}
            strategy = pred_config["ensemble_strategy"]
            if strategy not in valid_strategies:
                error = self._create_validation_error(
                    code=ConfigValidationErrorCodes.INVALID_ENUM_VALUE,
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.SYNTAX,
                    field_path="prediction_service.ensemble_strategy",
                    current_value=strategy,
                    expected_value=f"one of {valid_strategies}"
                )
                result.errors.append(error)

    def _create_validation_error(self, code: str, severity: ValidationSeverity, 
                               category: ValidationCategory, field_path: str,
                               current_value: Any, expected_value: Any) -> ValidationError:
        """Create structured validation error with guidance."""
        
        # Get guidance for this error code
        guidance = self.guidance.get_error_guidance(code)
        
        if guidance:
            title = guidance["title"]
            description = guidance["description"]
            env_var = f"{field_path.upper().replace('.', '_')}" if field_path else ""
            remediation = guidance["remediation_template"].format(
                field=field_path,
                current_value=current_value,
                expected_value=expected_value,
                env_var=env_var,
                expected_format=expected_value
            )
            documentation_url = guidance.get("documentation_url")
            examples = guidance.get("examples", [])
        else:
            title = f"Configuration validation failed for {field_path}"
            description = f"The value '{current_value}' is invalid for field '{field_path}'"
            remediation = f"Please correct the value for '{field_path}'"
            documentation_url = None
            examples = []
        
        return ValidationError(
            code=code,
            severity=severity,
            category=category,
            title=title,
            message=description,
            field_path=field_path,
            current_value=current_value,
            expected_value=expected_value,
            remediation=remediation,
            documentation_url=documentation_url,
            examples=examples
        )

    def _create_critical_error(self, code: str, title: str, message: str) -> ValidationError:
        """Create a critical validation error."""
        
        return ValidationError(
            code=code,
            severity=ValidationSeverity.CRITICAL,
            category=ValidationCategory.SCHEMA,
            title=title,
            message=message,
            remediation="Check configuration syntax and structure"
        )

    def generate_validation_report(self, result: ValidationResult, format: str = "text") -> str:
        """Generate formatted validation report."""
        
        if format == "json":
            return self._generate_json_report(result)
        elif format == "markdown":
            return self._generate_markdown_report(result)
        else:
            return self._generate_text_report(result)
    
    def _generate_text_report(self, result: ValidationResult) -> str:
        """Generate text format validation report."""
        
        lines = []
        lines.append("Configuration Validation Report")
        lines.append("=" * 40)
        lines.append(f"Status: {'PASSED' if result.is_valid else 'FAILED'}")
        lines.append(f"Validation Time: {result.validation_time:.3f}s")
        lines.append("")
        
        if result.errors:
            lines.append("ERRORS:")
            for error in result.errors:
                lines.append(f"  [{error.code}] {error.title}")
                lines.append(f"    Field: {error.field_path}")
                lines.append(f"    Issue: {error.message}")
                lines.append(f"    Fix: {error.remediation}")
                if error.documentation_url:
                    lines.append(f"    Docs: {error.documentation_url}")
                lines.append("")
        
        if result.warnings:
            lines.append("WARNINGS:")
            for warning in result.warnings:
                lines.append(f"  [{warning.code}] {warning.title}")
                lines.append(f"    Field: {warning.field_path}")
                lines.append(f"    Issue: {warning.message}")
                lines.append(f"    Recommendation: {warning.remediation}")
                lines.append("")
        
        if result.suggestions:
            lines.append("SUGGESTIONS:")
            for suggestion in result.suggestions:
                lines.append(f"  [{suggestion.code}] {suggestion.title}")
                lines.append(f"    Field: {suggestion.field_path}")
                lines.append(f"    Suggestion: {suggestion.remediation}")
                lines.append("")
        
        return "\n".join(lines)

    def _generate_json_report(self, result: ValidationResult) -> str:
        """Generate JSON format validation report."""
        
        def error_to_dict(error: ValidationError) -> dict[str, Any]:
            return {
                "code": error.code,
                "severity": error.severity,
                "category": error.category,
                "title": error.title,
                "message": error.message,
                "field_path": error.field_path,
                "current_value": str(error.current_value) if error.current_value is not None else None,
                "expected_value": str(error.expected_value) if error.expected_value is not None else None,
                "remediation": error.remediation,
                "documentation_url": error.documentation_url,
                "examples": error.examples
            }
        
        report = {
            "validation_result": {
                "is_valid": result.is_valid,
                "validation_time": result.validation_time,
                "validator_version": result.validator_version,
                "errors": [error_to_dict(e) for e in result.errors],
                "warnings": [error_to_dict(w) for w in result.warnings],
                "suggestions": [error_to_dict(s) for s in result.suggestions]
            }
        }
        
        return json.dumps(report, indent=2)

    def _generate_markdown_report(self, result: ValidationResult) -> str:
        """Generate Markdown format validation report."""
        
        lines = []
        lines.append("# Configuration Validation Report")
        lines.append("")
        lines.append(f"**Status:** {'✅ PASSED' if result.is_valid else '❌ FAILED'}")
        lines.append(f"**Validation Time:** {result.validation_time:.3f}s")
        lines.append("")
        
        if result.errors:
            lines.append("## ❌ Errors")
            lines.append("")
            for error in result.errors:
                lines.append(f"### [{error.code}] {error.title}")
                lines.append(f"- **Field:** `{error.field_path}`")
                lines.append(f"- **Issue:** {error.message}")
                lines.append(f"- **Fix:** {error.remediation}")
                if error.documentation_url:
                    lines.append(f"- **Documentation:** [{error.documentation_url}]({error.documentation_url})")
                lines.append("")
        
        if result.warnings:
            lines.append("## ⚠️ Warnings")
            lines.append("")
            for warning in result.warnings:
                lines.append(f"### [{warning.code}] {warning.title}")
                lines.append(f"- **Field:** `{warning.field_path}`")
                lines.append(f"- **Issue:** {warning.message}")
                lines.append(f"- **Recommendation:** {warning.remediation}")
                lines.append("")
        
        if result.suggestions:
            lines.append("## 💡 Suggestions")
            lines.append("")
            for suggestion in result.suggestions:
                lines.append(f"### [{suggestion.code}] {suggestion.title}")
                lines.append(f"- **Field:** `{suggestion.field_path}`")
                lines.append(f"- **Suggestion:** {suggestion.remediation}")
                lines.append("")
        
        return "\n".join(lines)


def validate_config(config: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    """Validate configuration and return results.

    Args:
        config: Configuration dictionary to validate

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = ConfigValidator(config)
    is_valid = validator.validate_all()
    return is_valid, validator.get_errors(), validator.get_warnings()


def validate_config_enhanced(config: dict[str, Any]) -> ValidationResult:
    """Enhanced validation with structured error reporting and guidance.
    
    This function replaces placeholder warnings with formal validation errors
    and provides comprehensive configuration guidance.

    Args:
        config: Configuration dictionary to validate

    Returns:
        ValidationResult with structured errors, warnings, and suggestions
    """
    validator = ConfigValidator(config)
    return validator.validate_with_formal_errors()


def check_for_hardcoded_values(file_path: str) -> list[str]:
    """Scan a Python file for potential hardcoded values that should be configurable.

    Args:
        file_path: Path to the Python file to scan

    Returns:
        List of potential issues found
    """
    issues = []

    # Patterns that might indicate hardcoded values
    patterns = [
        (r'"https?://[^"]+\.com[^"]*"', "Hardcoded URL"),
        (r'"wss?://[^"]+\.com[^"]*"', "Hardcoded WebSocket URL"),
        (r"timeout\s*=\s*\d+", "Hardcoded timeout value"),
        (r"max_retries\s*=\s*\d+", "Hardcoded retry count"),
        (r"sleep\(\s*\d+\s*\)", "Hardcoded sleep duration"),
        (r"range\(\s*\d+\s*,\s*\d+\s*\)", "Hardcoded range values"),
    ]

    try:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")

        for line_num, line in enumerate(content.split("\n"), 1):
            # Skip comments and docstrings
            stripped = line.strip()
            if stripped.startswith(("#", '"""', "'''")):
                continue

            for pattern, description in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(f"Line {line_num}: {description} - {line.strip()}")

    except Exception as e:
        issues.append(f"Error scanning file {file_path}: {e!s}")

    return issues