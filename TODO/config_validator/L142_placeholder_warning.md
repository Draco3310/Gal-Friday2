# Task: Replace placeholder warning message with formal validation error or configuration guidance

### 1. Context
- **File:** `gal_friday/utils/config_validator.py`
- **Line:** `142`
- **Keyword/Pattern:** `"Placeholder"`
- **Current State:** Placeholder warning message that needs formal validation error handling and configuration guidance

### 2. Problem Statement
The configuration validator contains placeholder warning messages that don't provide actionable guidance to users. This creates poor user experience and makes it difficult to debug configuration issues in production environments.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Validation Error Framework:** Structured error reporting with error codes and descriptions
2. **Build Configuration Guidance System:** Actionable guidance for fixing configuration issues
3. **Implement Error Categorization:** Categorize errors by severity and impact
4. **Add Remediation Suggestions:** Specific steps to resolve configuration problems
5. **Create Validation Reports:** Comprehensive validation reporting with multiple formats
6. **Build Help System:** Interactive help for configuration validation

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import logging
from pathlib import Path

class ValidationSeverity(str, Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUGGESTION = "suggestion"

class ValidationCategory(str, Enum):
    """Categories of validation issues"""
    SYNTAX = "syntax"
    SCHEMA = "schema"
    DEPENDENCY = "dependency"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"
    BEST_PRACTICE = "best_practice"

@dataclass
class ValidationError:
    """Structured validation error with guidance"""
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
    examples: List[Dict[str, Any]] = field(default_factory=list)
    related_errors: List[str] = field(default_factory=list)

@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    suggestions: List[ValidationError] = field(default_factory=list)
    validation_time: float = 0.0
    config_version: Optional[str] = None
    validator_version: str = "1.0.0"

class ConfigValidationErrorCodes:
    """Standard error codes for configuration validation"""
    
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
    
    # Performance warnings
    SUBOPTIMAL_SETTINGS = "CV301"
    RESOURCE_LIMITS = "CV302"
    
    # Best practice suggestions
    DEPRECATED_SETTING = "CV401"
    MISSING_OPTIONAL_FIELD = "CV402"
    INCONSISTENT_NAMING = "CV403"

class ConfigurationGuidance:
    """Provides guidance for configuration validation errors"""
    
    def __init__(self):
        self.error_guidance = self._initialize_error_guidance()
        self.field_documentation = self._initialize_field_documentation()
    
    def _initialize_error_guidance(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error code guidance"""
        
        return {
            ConfigValidationErrorCodes.MISSING_REQUIRED_FIELD: {
                "title": "Missing Required Configuration Field",
                "description": "A required configuration field is missing",
                "remediation_template": "Add the required field '{field}' to your configuration",
                "examples": [
                    {"field": "database_url", "value": "postgresql://user:pass@localhost:5432/db"},
                    {"field": "api_key", "value": "your-api-key-here"}
                ],
                "documentation_url": "https://docs.example.com/config/required-fields"
            },
            
            ConfigValidationErrorCodes.INVALID_DATA_TYPE: {
                "title": "Invalid Data Type",
                "description": "The value provided has an incorrect data type",
                "remediation_template": "Change the value of '{field}' from {current_type} to {expected_type}",
                "examples": [
                    {"field": "port", "correct": 8080, "incorrect": "8080"},
                    {"field": "enabled", "correct": True, "incorrect": "true"}
                ],
                "documentation_url": "https://docs.example.com/config/data-types"
            },
            
            ConfigValidationErrorCodes.INVALID_URL_FORMAT: {
                "title": "Invalid URL Format",
                "description": "The URL format is incorrect or malformed",
                "remediation_template": "Correct the URL format for '{field}'. Expected format: {expected_format}",
                "examples": [
                    {"field": "api_endpoint", "correct": "https://api.example.com/v1", "incorrect": "api.example.com"},
                    {"field": "webhook_url", "correct": "https://webhook.site/abc123", "incorrect": "webhook.site"}
                ],
                "documentation_url": "https://docs.example.com/config/url-format"
            },
            
            ConfigValidationErrorCodes.WEAK_CREDENTIALS: {
                "title": "Weak Credentials Detected",
                "description": "The configured credentials do not meet security requirements",
                "remediation_template": "Strengthen the credentials for '{field}'. Use at least 12 characters with mixed case, numbers, and symbols",
                "examples": [
                    {"field": "admin_password", "weak": "password123", "strong": "P@ssw0rd!2023#Str0ng"},
                    {"field": "api_secret", "weak": "secret", "strong": "aB3$dF6&hJ9*kL2@nP5^"}
                ],
                "documentation_url": "https://docs.example.com/security/credentials"
            }
        }
    
    def _initialize_field_documentation(self) -> Dict[str, Dict[str, Any]]:
        """Initialize field-specific documentation"""
        
        return {
            "database_url": {
                "description": "Connection string for the primary database",
                "format": "postgresql://username:password@host:port/database",
                "required": True,
                "examples": ["postgresql://trader:secret@localhost:5432/trading"]
            },
            
            "api_port": {
                "description": "Port number for the API server",
                "type": "integer",
                "range": [1024, 65535],
                "default": 8080,
                "examples": [8080, 8443, 9000]
            },
            
            "log_level": {
                "description": "Logging level for the application",
                "type": "string",
                "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "default": "INFO",
                "examples": ["INFO", "DEBUG"]
            }
        }
    
    def get_error_guidance(self, error_code: str) -> Optional[Dict[str, Any]]:
        """Get guidance for specific error code"""
        return self.error_guidance.get(error_code)
    
    def get_field_documentation(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Get documentation for specific field"""
        return self.field_documentation.get(field_name)

class EnhancedConfigValidator:
    """Enhanced configuration validator with formal error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.guidance = ConfigurationGuidance()
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'error_counts': {},
            'most_common_errors': []
        }
    
    def validate_configuration(self, config: Dict[str, Any], 
                             schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate configuration with formal error reporting
        Replace placeholder warning with structured validation
        """
        
        import time
        start_time = time.time()
        
        try:
            self.validation_stats['total_validations'] += 1
            
            result = ValidationResult()
            
            # Perform various validation checks
            self._validate_required_fields(config, result)
            self._validate_data_types(config, result)
            self._validate_field_values(config, result)
            self._validate_dependencies(config, result)
            self._validate_security(config, result)
            self._validate_performance(config, result)
            self._check_best_practices(config, result)
            
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
    
    def _validate_required_fields(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate required configuration fields"""
        
        required_fields = [
            'database_url',
            'api_port',
            'log_level'
        ]
        
        for field in required_fields:
            if field not in config:
                error = self._create_validation_error(
                    code=ConfigValidationErrorCodes.MISSING_REQUIRED_FIELD,
                    severity=ValidationSeverity.CRITICAL,
                    category=ValidationCategory.SCHEMA,
                    field_path=field,
                    current_value=None,
                    expected_value="required"
                )
                result.errors.append(error)
    
    def _validate_data_types(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate data types of configuration values"""
        
        type_expectations = {
            'api_port': int,
            'debug_enabled': bool,
            'max_connections': int,
            'timeout_seconds': float,
            'allowed_origins': list
        }
        
        for field, expected_type in type_expectations.items():
            if field in config:
                actual_value = config[field]
                if not isinstance(actual_value, expected_type):
                    error = self._create_validation_error(
                        code=ConfigValidationErrorCodes.INVALID_DATA_TYPE,
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.SCHEMA,
                        field_path=field,
                        current_value=actual_value,
                        expected_value=expected_type.__name__
                    )
                    result.errors.append(error)
    
    def _validate_field_values(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate specific field values"""
        
        # Validate port numbers
        if 'api_port' in config:
            port = config['api_port']
            if isinstance(port, int) and not (1024 <= port <= 65535):
                error = self._create_validation_error(
                    code=ConfigValidationErrorCodes.INVALID_PORT_NUMBER,
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.SYNTAX,
                    field_path='api_port',
                    current_value=port,
                    expected_value="1024-65535"
                )
                result.errors.append(error)
        
        # Validate URLs
        url_fields = ['database_url', 'api_endpoint', 'webhook_url']
        for field in url_fields:
            if field in config:
                url = config[field]
                if isinstance(url, str) and not self._is_valid_url(url):
                    error = self._create_validation_error(
                        code=ConfigValidationErrorCodes.INVALID_URL_FORMAT,
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.SYNTAX,
                        field_path=field,
                        current_value=url,
                        expected_value="valid URL format"
                    )
                    result.errors.append(error)
    
    def _validate_security(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate security-related configuration"""
        
        # Check for weak passwords
        password_fields = ['admin_password', 'database_password', 'secret_key']
        for field in password_fields:
            if field in config:
                password = config[field]
                if isinstance(password, str) and self._is_weak_password(password):
                    error = self._create_validation_error(
                        code=ConfigValidationErrorCodes.WEAK_CREDENTIALS,
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.SECURITY,
                        field_path=field,
                        current_value="[REDACTED]",
                        expected_value="strong password"
                    )
                    result.warnings.append(error)
        
        # Check for insecure protocols
        if 'api_endpoint' in config:
            url = config['api_endpoint']
            if isinstance(url, str) and url.startswith('http://'):
                error = self._create_validation_error(
                    code=ConfigValidationErrorCodes.INSECURE_PROTOCOL,
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.SECURITY,
                    field_path='api_endpoint',
                    current_value=url,
                    expected_value="HTTPS URL"
                )
                result.warnings.append(error)
    
    def _create_validation_error(self, code: str, severity: ValidationSeverity, 
                               category: ValidationCategory, field_path: str,
                               current_value: Any, expected_value: Any) -> ValidationError:
        """
        Create structured validation error with guidance
        Replace placeholder warning message with formal error
        """
        
        # Get guidance for this error code
        guidance = self.guidance.get_error_guidance(code)
        
        if guidance:
            title = guidance["title"]
            description = guidance["description"]
            remediation = guidance["remediation_template"].format(
                field=field_path,
                current_value=current_value,
                expected_value=expected_value,
                current_type=type(current_value).__name__ if current_value is not None else "None",
                expected_type=expected_value
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
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None
    
    def _is_weak_password(self, password: str) -> bool:
        """Check if password is weak"""
        if len(password) < 8:
            return True
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return not (has_upper and has_lower and has_digit and has_special)
    
    def generate_validation_report(self, result: ValidationResult, format: str = "text") -> str:
        """Generate formatted validation report"""
        
        if format == "json":
            return self._generate_json_report(result)
        elif format == "markdown":
            return self._generate_markdown_report(result)
        else:
            return self._generate_text_report(result)
    
    def _generate_text_report(self, result: ValidationResult) -> str:
        """Generate text format validation report"""
        
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
                lines.append("")
        
        if result.warnings:
            lines.append("WARNINGS:")
            for warning in result.warnings:
                lines.append(f"  [{warning.code}] {warning.title}")
                lines.append(f"    Field: {warning.field_path}")
                lines.append(f"    Issue: {warning.message}")
                lines.append(f"    Recommendation: {warning.remediation}")
                lines.append("")
        
        return "\n".join(lines)
```

#### c. Key Considerations & Dependencies
- **User Experience:** Clear, actionable error messages with specific guidance
- **Documentation:** Links to relevant documentation and examples
- **Categorization:** Proper categorization by severity and type
- **Remediation:** Specific steps to fix configuration issues

### 4. Acceptance Criteria
- [ ] Structured validation error framework with error codes
- [ ] Formal error messages replace placeholder warnings
- [ ] Comprehensive configuration guidance system
- [ ] Error categorization by severity and impact
- [ ] Actionable remediation suggestions for each error
- [ ] Field-specific documentation and examples
- [ ] Multiple validation report formats (text, JSON, markdown)
- [ ] Security validation for credentials and protocols
- [ ] Performance validation for resource settings
- [ ] Best practice suggestions for optimal configuration
- [ ] Placeholder warning messages completely replaced with formal validation system 