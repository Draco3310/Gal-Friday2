# Minor Implementation Details Implementation Design

**File**: `/gal_friday/dal/models/position_adjustment.py`
- **Line 23**: Schema version assumption comment
- **Issue**: Database schema version handling not properly implemented

**File**: `/gal_friday/utils/__init__.py`
- **Line 34**: `# For now, let the handlers manage the default if it remains None`
- **Issue**: Exception handler default configuration deferred instead of properly implemented

**File**: `/gal_friday/portfolio/position_manager.py`
- **Line 433**: `raise # Re-raise for now`
- **Line 451**: `# For now, keeping if the internal logic still uses them...`
- **Issue**: Error handling and legacy code management not properly resolved

**Impact**: Minor implementation gaps that could lead to production issues and maintenance complexity

## Overview
Various minor implementation details have been deferred with "for now" comments, creating technical debt and potential production issues. This design implements proper solutions for database schema management, exception handling configuration, and portfolio management error scenarios.

## Architecture Design

### 1. Current Minor Implementation Issues

```
Minor Implementation Problems:
├── Database Schema Management
│   ├── Hardcoded schema assumptions
│   ├── No version migration support
│   ├── Missing schema validation
│   └── Poor database evolution strategy
├── Exception Handler Configuration
│   ├── Deferred default initialization
│   ├── Incomplete configuration validation
│   ├── Missing type safety
│   └── Poor error handling strategy
├── Portfolio Management Issues
│   ├── Generic error re-raising
│   ├── Legacy code uncertainty
│   ├── Incomplete error recovery
│   └── Poor exception classification
└── Technical Debt Impact
    ├── Increased maintenance burden
    ├── Potential runtime failures
    ├── Poor error diagnostics
    └── Reduced system reliability
```

### 2. Production-Ready Implementation Architecture

```
Enterprise Implementation Solutions:
├── Advanced Database Schema Management
│   ├── Schema Version Control
│   │   ├── Automated migration framework
│   │   ├── Version compatibility checking
│   │   ├── Rollback capabilities
│   │   └── Schema validation pipeline
│   ├── Dynamic Schema Adaptation
│   │   ├── Runtime schema detection
│   │   ├── Backward compatibility layers
│   │   ├── Forward compatibility planning
│   │   └── Schema evolution tracking
│   ├── Database Health Monitoring
│   │   ├── Schema integrity checks
│   │   ├── Performance monitoring
│   │   ├── Constraint validation
│   │   └── Index optimization
│   └── Production Schema Operations
│       ├── Zero-downtime migrations
│       ├── Blue-green deployments
│       ├── Schema change approvals
│       └── Automated testing frameworks
├── Comprehensive Exception Handler Framework
│   ├── Advanced Configuration Management
│   │   ├── Type-safe configuration
│   │   ├── Runtime validation
│   │   ├── Environment-specific settings
│   │   └── Hot-reload capabilities
│   ├── Intelligent Default Handling
│   │   ├── Context-aware defaults
│   │   ├── Dynamic default generation
│   │   ├── Fallback strategies
│   │   └── Configuration inheritance
│   ├── Exception Classification System
│   │   ├── Error severity levels
│   │   ├── Recovery strategies
│   │   ├── Escalation policies
│   │   └── Business impact assessment
│   └── Monitoring & Alerting
│       ├── Exception pattern detection
│       ├── Performance impact analysis
│       ├── Automated remediation
│       └── Trend analysis
├── Production Portfolio Management
│   ├── Sophisticated Error Handling
│   │   ├── Error categorization
│   │   ├── Recovery mechanisms
│   │   ├── Compensation strategies
│   │   └── State consistency guarantees
│   ├── Legacy Code Management
│   │   ├── Deprecation strategies
│   │   ├── Migration planning
│   │   ├── Compatibility layers
│   │   └── Technical debt tracking
│   ├── Resilience Engineering
│   │   ├── Circuit breaker patterns
│   │   ├── Bulkhead isolation
│   │   ├── Graceful degradation
│   │   └── Self-healing mechanisms
│   └── Operational Excellence
│       ├── Performance monitoring
│       ├── Resource optimization
│       ├── Capacity planning
│       └── Predictive maintenance
└── Quality Assurance Framework
    ├── Automated Testing
    │   ├── Unit test coverage
    │   ├── Integration testing
    │   ├── End-to-end validation
    │   └── Performance testing
    ├── Code Quality Management
    │   ├── Static analysis
    │   ├── Security scanning
    │   ├── Complexity metrics
    │   └── Technical debt tracking
    ├── Production Readiness
    │   ├── Performance benchmarks
    │   ├── Scalability validation
    │   ├── Security assessment
    │   └── Compliance verification
    └── Continuous Improvement
        ├── Metrics-driven optimization
        ├── Feedback loop integration
        ├── Best practice enforcement
        └── Knowledge management
```

### 3. Key Features

1. **Schema Evolution**: Automated database schema management with version control
2. **Smart Configuration**: Intelligent exception handler configuration with type safety
3. **Robust Error Handling**: Comprehensive error classification and recovery
4. **Technical Debt Management**: Systematic approach to legacy code and improvements
5. **Production Readiness**: Enterprise-grade implementation with monitoring
6. **Quality Assurance**: Comprehensive testing and validation frameworks

## Implementation Plan

### Phase 1: Enhanced Database Schema Management

**File**: `/gal_friday/dal/models/position_adjustment.py`
**Target Line**: Line 23 - Replace schema assumptions with proper version management

```python
"""
Production-ready database schema management with version control and migration support.

This module provides enterprise-grade schema management replacing hardcoded
assumptions with dynamic version detection and automated migration capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import uuid

from sqlalchemy import text, inspect, MetaData
from sqlalchemy.orm import Session
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

from gal_friday.logger_service import LoggerService


class SchemaVersionStatus(Enum):
    """Schema version status enumeration."""
    CURRENT = "current"
    OUTDATED = "outdated"
    AHEAD = "ahead"
    UNKNOWN = "unknown"
    INCOMPATIBLE = "incompatible"


@dataclass
class SchemaVersionInfo:
    """Schema version information container."""
    current_version: str
    target_version: str
    status: SchemaVersionStatus
    migration_required: bool
    migration_path: List[str]
    compatibility_level: str
    last_migration_date: Optional[datetime]
    schema_hash: str


class SchemaVersionManager:
    """
    Production-ready schema version management.
    
    Replaces hardcoded schema assumptions with dynamic version detection
    and automated migration capabilities.
    """
    
    def __init__(self, db_session: Session, logger: LoggerService):
        self.db_session = db_session
        self.logger = logger
        self.metadata = MetaData()
        self._schema_cache: Dict[str, Any] = {}
        
    async def get_schema_version_info(self) -> SchemaVersionInfo:
        """
        Get comprehensive schema version information.
        
        This replaces the hardcoded schema assumptions with dynamic detection
        and validation of the current database schema state.
        """
        try:
            # Detect current schema version
            current_version = await self._detect_current_schema_version()
            
            # Get target version from application
            target_version = await self._get_target_schema_version()
            
            # Determine version status
            status = await self._determine_version_status(current_version, target_version)
            
            # Calculate migration requirements
            migration_required = status in [SchemaVersionStatus.OUTDATED, SchemaVersionStatus.INCOMPATIBLE]
            migration_path = await self._calculate_migration_path(current_version, target_version)
            
            # Get compatibility information
            compatibility_level = await self._assess_compatibility(current_version, target_version)
            
            # Get last migration date
            last_migration_date = await self._get_last_migration_date()
            
            # Calculate schema hash for integrity verification
            schema_hash = await self._calculate_schema_hash()
            
            version_info = SchemaVersionInfo(
                current_version=current_version,
                target_version=target_version,
                status=status,
                migration_required=migration_required,
                migration_path=migration_path,
                compatibility_level=compatibility_level,
                last_migration_date=last_migration_date,
                schema_hash=schema_hash
            )
            
            self.logger.info(
                f"Schema version info: {current_version} -> {target_version} ({status.value})",
                extra={
                    "current_version": current_version,
                    "target_version": target_version,
                    "status": status.value,
                    "migration_required": migration_required
                }
            )
            
            return version_info
            
        except Exception as e:
            self.logger.error(f"Failed to get schema version info: {e}", exc_info=True)
            raise SchemaVersionError(f"Schema version detection failed: {e}") from e
    
    async def _detect_current_schema_version(self) -> str:
        """Detect current database schema version."""
        try:
            # Check for Alembic version table
            inspector = inspect(self.db_session.bind)
            tables = inspector.get_table_names()
            
            if 'alembic_version' in tables:
                # Get version from Alembic
                result = self.db_session.execute(
                    text("SELECT version_num FROM alembic_version ORDER BY version_num DESC LIMIT 1")
                ).fetchone()
                
                if result:
                    return result[0]
            
            # Fallback to schema introspection
            return await self._introspect_schema_version()
            
        except Exception as e:
            self.logger.warning(f"Failed to detect schema version: {e}")
            return "unknown"
    
    async def _introspect_schema_version(self) -> str:
        """Introspect schema version from table structure."""
        try:
            inspector = inspect(self.db_session.bind)
            
            # Check for specific tables and columns that indicate version
            version_indicators = {
                "position_adjustments": ["adjustment_id", "reconciliation_id", "created_at"],
                "positions": ["position_id", "trading_pair", "last_updated_at"],
                "trades": ["trade_id", "realized_pnl", "exit_timestamp"]
            }
            
            schema_signature = []
            
            for table_name, expected_columns in version_indicators.items():
                if table_name in inspector.get_table_names():
                    columns = [col['name'] for col in inspector.get_columns(table_name)]
                    matching_columns = [col for col in expected_columns if col in columns]
                    schema_signature.append(f"{table_name}:{len(matching_columns)}")
            
            # Generate version based on schema signature
            signature_hash = hash('|'.join(sorted(schema_signature)))
            return f"introspected_{abs(signature_hash) % 10000:04d}"
            
        except Exception as e:
            self.logger.error(f"Schema introspection failed: {e}")
            return "unknown"
    
    async def _get_target_schema_version(self) -> str:
        """Get target schema version from application configuration."""
        try:
            # Read from Alembic configuration
            alembic_cfg = Config("alembic.ini")
            script_dir = ScriptDirectory.from_config(alembic_cfg)
            
            # Get head revision
            head_revision = script_dir.get_current_head()
            if head_revision:
                return head_revision
            
            # Fallback to application version
            return await self._get_application_schema_version()
            
        except Exception as e:
            self.logger.warning(f"Failed to get target version: {e}")
            return "latest"
    
    async def _determine_version_status(self, current: str, target: str) -> SchemaVersionStatus:
        """Determine schema version status."""
        if current == "unknown" or target == "unknown":
            return SchemaVersionStatus.UNKNOWN
        
        if current == target:
            return SchemaVersionStatus.CURRENT
        
        # Check if current version is ahead of target
        if await self._is_version_ahead(current, target):
            return SchemaVersionStatus.AHEAD
        
        # Check compatibility
        if await self._are_versions_compatible(current, target):
            return SchemaVersionStatus.OUTDATED
        else:
            return SchemaVersionStatus.INCOMPATIBLE
    
    async def validate_schema_integrity(self) -> Dict[str, Any]:
        """Validate database schema integrity."""
        try:
            validation_results = {
                "tables_exist": [],
                "missing_tables": [],
                "column_mismatches": [],
                "constraint_issues": [],
                "index_problems": [],
                "overall_status": "unknown"
            }
            
            inspector = inspect(self.db_session.bind)
            existing_tables = set(inspector.get_table_names())
            
            # Expected schema definition
            expected_schema = await self._get_expected_schema_definition()
            
            # Validate table existence
            for table_name in expected_schema.get("tables", []):
                if table_name in existing_tables:
                    validation_results["tables_exist"].append(table_name)
                else:
                    validation_results["missing_tables"].append(table_name)
            
            # Validate columns for existing tables
            for table_name in validation_results["tables_exist"]:
                column_issues = await self._validate_table_columns(table_name, expected_schema)
                if column_issues:
                    validation_results["column_mismatches"].extend(column_issues)
            
            # Validate constraints
            constraint_issues = await self._validate_schema_constraints(expected_schema)
            validation_results["constraint_issues"] = constraint_issues
            
            # Validate indexes
            index_issues = await self._validate_schema_indexes(expected_schema)
            validation_results["index_problems"] = index_issues
            
            # Determine overall status
            if (not validation_results["missing_tables"] and 
                not validation_results["column_mismatches"] and
                not validation_results["constraint_issues"] and
                not validation_results["index_problems"]):
                validation_results["overall_status"] = "valid"
            elif validation_results["missing_tables"]:
                validation_results["overall_status"] = "critical"
            else:
                validation_results["overall_status"] = "warnings"
            
            self.logger.info(
                f"Schema integrity validation completed: {validation_results['overall_status']}",
                extra=validation_results
            )
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Schema integrity validation failed: {e}", exc_info=True)
            return {"overall_status": "error", "error": str(e)}
    
    async def perform_safe_migration(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """Perform safe database migration with rollback capabilities."""
        try:
            migration_results = {
                "success": False,
                "target_version": target_version or "latest",
                "applied_migrations": [],
                "rollback_point": None,
                "migration_time": None,
                "pre_migration_backup": None
            }
            
            start_time = datetime.now(timezone.utc)
            
            # Create backup before migration
            backup_info = await self._create_pre_migration_backup()
            migration_results["pre_migration_backup"] = backup_info
            
            # Get current version as rollback point
            current_version = await self._detect_current_schema_version()
            migration_results["rollback_point"] = current_version
            
            # Validate migration safety
            safety_check = await self._validate_migration_safety(current_version, target_version)
            if not safety_check["safe"]:
                raise MigrationError(f"Migration safety check failed: {safety_check['reasons']}")
            
            # Perform migration
            applied_migrations = await self._execute_migration(target_version)
            migration_results["applied_migrations"] = applied_migrations
            
            # Validate post-migration state
            post_migration_validation = await self.validate_schema_integrity()
            if post_migration_validation["overall_status"] not in ["valid", "warnings"]:
                # Rollback on validation failure
                await self._perform_rollback(current_version)
                raise MigrationError("Post-migration validation failed, rolled back")
            
            migration_results["success"] = True
            migration_results["migration_time"] = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            self.logger.info(
                f"Migration completed successfully: {current_version} -> {target_version}",
                extra=migration_results
            )
            
            return migration_results
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}", exc_info=True)
            migration_results["success"] = False
            migration_results["error"] = str(e)
            return migration_results


class SchemaVersionError(Exception):
    """Schema version management error."""
    pass


class MigrationError(Exception):
    """Database migration error."""
    pass


# Enhanced PositionAdjustment model with proper schema management
class EnhancedPositionAdjustment(Base):
    """
    Enhanced position adjustment model with proper schema version management.
    
    This replaces hardcoded schema assumptions with dynamic version detection
    and proper database evolution support.
    """
    
    __tablename__ = "position_adjustments"
    
    # Use schema manager for version-aware field definitions
    adjustment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.uuid_generate_v4(),
    )
    
    reconciliation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("reconciliation_events.reconciliation_id"),
        nullable=False,
        index=True,
    )
    
    @classmethod
    async def validate_schema_compatibility(cls, db_session: Session) -> bool:
        """Validate that the current schema is compatible with this model."""
        try:
            schema_manager = SchemaVersionManager(db_session, LoggerService())
            version_info = await schema_manager.get_schema_version_info()
            
            # Check if current schema supports this model
            return version_info.compatibility_level in ["full", "partial"]
            
        except Exception:
            return False
    
    @classmethod
    async def get_schema_migration_requirements(cls, db_session: Session) -> Optional[Dict[str, Any]]:
        """Get schema migration requirements for this model."""
        try:
            schema_manager = SchemaVersionManager(db_session, LoggerService())
            version_info = await schema_manager.get_schema_version_info()
            
            if version_info.migration_required:
                return {
                    "current_version": version_info.current_version,
                    "target_version": version_info.target_version,
                    "migration_path": version_info.migration_path,
                    "estimated_downtime": await cls._estimate_migration_downtime(version_info)
                }
            
            return None
            
        except Exception as e:
            logging.error(f"Failed to get migration requirements: {e}")
            return None
```

### Phase 2: Advanced Exception Handler Configuration

**File**: `/gal_friday/utils/__init__.py`
**Target Line**: Line 34 - Replace deferred initialization with proper configuration

```python
"""
Production-ready exception handler configuration with type safety and intelligent defaults.

This module provides enterprise-grade exception handling configuration replacing
deferred initialization with proper type-safe configuration management.
"""

import logging
from typing import TypeVar, Generic, Optional, Dict, Any, Callable, Union, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import functools
import inspect

T = TypeVar('T')


class ExceptionSeverity(Enum):
    """Exception severity levels for classification."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ExceptionHandlingStrategy(Enum):
    """Exception handling strategies."""
    RAISE = "raise"
    LOG_AND_CONTINUE = "log_and_continue"
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class ExceptionRule:
    """Exception handling rule configuration."""
    exception_types: Tuple[Type[Exception], ...]
    severity: ExceptionSeverity
    strategy: ExceptionHandlingStrategy
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_value: Any = None
    custom_handler: Optional[Callable] = None
    alert_threshold: int = 5


@dataclass
class ExceptionHandlerConfig(Generic[T]):
    """
    Production-ready exception handler configuration.
    
    This replaces the deferred "for now" approach with comprehensive
    type-safe configuration with intelligent defaults.
    """
    
    # Core configuration
    exception_rules: Dict[str, ExceptionRule] = field(default_factory=dict)
    default_exception_types: Tuple[Type[Exception], ...] = field(default_factory=lambda: (Exception,))
    default_severity: ExceptionSeverity = ExceptionSeverity.MEDIUM
    default_strategy: ExceptionHandlingStrategy = ExceptionHandlingStrategy.RAISE
    
    # Advanced configuration
    enable_context_aware_handling: bool = True
    enable_performance_monitoring: bool = True
    enable_pattern_detection: bool = True
    enable_automatic_escalation: bool = True
    
    # Monitoring and alerting
    alert_channels: Dict[str, Any] = field(default_factory=dict)
    metrics_collector: Optional[Callable] = None
    error_reporter: Optional[Callable] = None
    
    # Environment-specific settings
    environment: str = "production"
    debug_mode: bool = False
    log_level: int = logging.WARNING
    
    def __post_init__(self):
        """Initialize configuration with intelligent defaults and validation."""
        # This replaces the "For now, let the handlers manage the default" approach
        # with proper initialization and validation
        
        if not self.exception_rules:
            self._initialize_default_rules()
        
        self._validate_configuration()
        self._setup_context_aware_defaults()
    
    def _initialize_default_rules(self) -> None:
        """Initialize intelligent default exception rules."""
        # Critical system exceptions
        self.exception_rules["critical_system"] = ExceptionRule(
            exception_types=(SystemExit, KeyboardInterrupt, MemoryError),
            severity=ExceptionSeverity.CRITICAL,
            strategy=ExceptionHandlingStrategy.RAISE,
            max_retries=0
        )
        
        # Database and network exceptions
        self.exception_rules["database_network"] = ExceptionRule(
            exception_types=(ConnectionError, TimeoutError),
            severity=ExceptionSeverity.HIGH,
            strategy=ExceptionHandlingStrategy.RETRY,
            max_retries=3,
            retry_delay=2.0
        )
        
        # Business logic exceptions
        self.exception_rules["business_logic"] = ExceptionRule(
            exception_types=(ValueError, TypeError, AttributeError),
            severity=ExceptionSeverity.MEDIUM,
            strategy=ExceptionHandlingStrategy.LOG_AND_CONTINUE,
            max_retries=1
        )
        
        # General exceptions
        self.exception_rules["general"] = ExceptionRule(
            exception_types=self.default_exception_types,
            severity=self.default_severity,
            strategy=self.default_strategy,
            max_retries=1
        )
    
    def _validate_configuration(self) -> None:
        """Validate configuration consistency and completeness."""
        # Validate exception rules
        for rule_name, rule in self.exception_rules.items():
            if not rule.exception_types:
                raise ValueError(f"Exception rule '{rule_name}' must specify exception types")
            
            if rule.max_retries < 0:
                raise ValueError(f"Exception rule '{rule_name}' max_retries must be non-negative")
            
            if rule.retry_delay < 0:
                raise ValueError(f"Exception rule '{rule_name}' retry_delay must be non-negative")
        
        # Validate environment-specific settings
        if self.environment not in ["development", "testing", "staging", "production"]:
            raise ValueError(f"Invalid environment: {self.environment}")
    
    def _setup_context_aware_defaults(self) -> None:
        """Setup context-aware default configurations."""
        if self.environment == "development":
            self.debug_mode = True
            self.log_level = logging.DEBUG
            self.default_strategy = ExceptionHandlingStrategy.RAISE
        
        elif self.environment == "production":
            self.debug_mode = False
            self.log_level = logging.WARNING
            self.enable_performance_monitoring = True
            self.enable_automatic_escalation = True
    
    def get_rule_for_exception(self, exception: Exception) -> ExceptionRule:
        """Get the most specific rule for a given exception."""
        for rule in self.exception_rules.values():
            if isinstance(exception, rule.exception_types):
                return rule
        
        # Fallback to general rule
        return self.exception_rules.get("general", ExceptionRule(
            exception_types=(Exception,),
            severity=self.default_severity,
            strategy=self.default_strategy
        ))


class IntelligentExceptionHandler:
    """
    Intelligent exception handler with advanced configuration management.
    
    This provides enterprise-grade exception handling with context awareness,
    pattern detection, and automatic optimization.
    """
    
    def __init__(self, config: ExceptionHandlerConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.exception_patterns: Dict[str, int] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self.circuit_breakers: Dict[str, Any] = {}
    
    def handle_exception(
        self, 
        exception: Exception, 
        context: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None
    ) -> Any:
        """
        Handle exception with intelligent strategy selection.
        
        This replaces simple re-raising with comprehensive exception management
        including context awareness and automatic strategy optimization.
        """
        try:
            # Get appropriate rule for this exception
            rule = self.config.get_rule_for_exception(exception)
            
            # Update pattern tracking
            self._update_exception_patterns(exception, operation_name)
            
            # Apply context-aware adjustments
            if self.config.enable_context_aware_handling and context:
                rule = self._adjust_rule_for_context(rule, context)
            
            # Execute handling strategy
            return self._execute_handling_strategy(exception, rule, context, operation_name)
            
        except Exception as handler_error:
            self.logger.error(
                f"Exception handler failed: {handler_error}",
                exc_info=True,
                extra={"original_exception": str(exception)}
            )
            # Fallback to raising original exception
            raise exception from handler_error
    
    def _execute_handling_strategy(
        self, 
        exception: Exception, 
        rule: ExceptionRule, 
        context: Optional[Dict[str, Any]], 
        operation_name: Optional[str]
    ) -> Any:
        """Execute the determined handling strategy."""
        strategy_map = {
            ExceptionHandlingStrategy.RAISE: self._strategy_raise,
            ExceptionHandlingStrategy.LOG_AND_CONTINUE: self._strategy_log_and_continue,
            ExceptionHandlingStrategy.RETRY: self._strategy_retry,
            ExceptionHandlingStrategy.FALLBACK: self._strategy_fallback,
            ExceptionHandlingStrategy.CIRCUIT_BREAKER: self._strategy_circuit_breaker
        }
        
        strategy_func = strategy_map.get(rule.strategy, self._strategy_raise)
        return strategy_func(exception, rule, context, operation_name)
    
    def _strategy_raise(self, exception: Exception, rule: ExceptionRule, context: Any, operation_name: str) -> None:
        """Strategy: Raise the exception."""
        self.logger.error(
            f"Exception raised: {exception}",
            exc_info=True,
            extra={
                "severity": rule.severity.value,
                "operation": operation_name,
                "context": context
            }
        )
        raise exception
    
    def _strategy_log_and_continue(self, exception: Exception, rule: ExceptionRule, context: Any, operation_name: str) -> Any:
        """Strategy: Log exception and continue with fallback value."""
        self.logger.warning(
            f"Exception handled with fallback: {exception}",
            extra={
                "severity": rule.severity.value,
                "operation": operation_name,
                "fallback_value": rule.fallback_value
            }
        )
        return rule.fallback_value
    
    def _update_exception_patterns(self, exception: Exception, operation_name: Optional[str]) -> None:
        """Update exception pattern tracking for optimization."""
        if self.config.enable_pattern_detection:
            pattern_key = f"{type(exception).__name__}:{operation_name or 'unknown'}"
            self.exception_patterns[pattern_key] = self.exception_patterns.get(pattern_key, 0) + 1
            
            # Check for escalation thresholds
            if self.config.enable_automatic_escalation:
                self._check_escalation_thresholds(pattern_key)


def enhanced_handle_exceptions(
    logger: logging.Logger,
    config: Optional[ExceptionHandlerConfig[T]] = None,
    operation_name: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Enhanced exception handling decorator with intelligent configuration.
    
    This replaces the deferred configuration approach with proper
    type-safe configuration and intelligent default management.
    """
    
    # Use intelligent defaults if no config provided
    if config is None:
        config = ExceptionHandlerConfig[T]()
    
    # Create intelligent handler
    handler = IntelligentExceptionHandler(config, logger)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Enhance context with function information
                enhanced_context = context or {}
                enhanced_context.update({
                    "function_name": func.__name__,
                    "function_module": func.__module__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                })
                
                return handler.handle_exception(
                    e, 
                    enhanced_context, 
                    operation_name or func.__name__
                )
        
        return wrapper
    
    return decorator


# Example usage with proper configuration
def create_production_exception_config() -> ExceptionHandlerConfig:
    """Create production-ready exception configuration."""
    return ExceptionHandlerConfig(
        environment="production",
        enable_context_aware_handling=True,
        enable_performance_monitoring=True,
        enable_pattern_detection=True,
        enable_automatic_escalation=True,
        alert_channels={
            "slack": {"webhook_url": "https://hooks.slack.com/..."},
            "email": {"recipients": ["ops@company.com"]},
            "pagerduty": {"service_key": "..."}
        }
    )
```

### Phase 3: Production Portfolio Management Error Handling

**File**: `/gal_friday/portfolio/position_manager.py`
**Target Lines**: Lines 433 and 451 - Replace basic error handling with comprehensive management

```python
"""
Production-ready portfolio management with comprehensive error handling and recovery.

This module provides enterprise-grade portfolio management replacing basic
error handling with sophisticated classification and recovery mechanisms.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from decimal import Decimal
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import asyncio
import uuid

from gal_friday.logger_service import LoggerService


class PortfolioErrorCategory(Enum):
    """Portfolio error categories for classification."""
    DATABASE_ERROR = "database_error"
    POSITION_CONSISTENCY = "position_consistency"
    CALCULATION_ERROR = "calculation_error"
    VALIDATION_ERROR = "validation_error"
    EXTERNAL_SERVICE = "external_service"
    CONCURRENCY_ISSUE = "concurrency_issue"
    CONFIGURATION_ERROR = "configuration_error"


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    IMMEDIATE_RETRY = "immediate_retry"
    DELAYED_RETRY = "delayed_retry"
    COMPENSATE = "compensate"
    ISOLATE = "isolate"
    ESCALATE = "escalate"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class PortfolioError:
    """Portfolio error information container."""
    error_id: str
    category: PortfolioErrorCategory
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    original_exception: Exception
    context: Dict[str, Any]
    timestamp: datetime
    trading_pair: Optional[str] = None
    position_id: Optional[str] = None
    operation: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def should_retry(self) -> bool:
        """Determine if error should be retried."""
        return (self.retry_count < self.max_retries and 
                self.recovery_strategy in [RecoveryStrategy.IMMEDIATE_RETRY, RecoveryStrategy.DELAYED_RETRY])


class PortfolioErrorHandler:
    """
    Comprehensive portfolio error handling and recovery system.
    
    This replaces basic error re-raising with sophisticated error classification,
    recovery mechanisms, and state consistency guarantees.
    """
    
    def __init__(self, logger: LoggerService):
        self.logger = logger
        self.error_history: List[PortfolioError] = []
        self.circuit_breakers: Dict[str, Any] = {}
        self.compensation_handlers: Dict[PortfolioErrorCategory, Callable] = {}
        self._setup_compensation_handlers()
    
    def _setup_compensation_handlers(self) -> None:
        """Setup compensation handlers for different error categories."""
        self.compensation_handlers = {
            PortfolioErrorCategory.DATABASE_ERROR: self._compensate_database_error,
            PortfolioErrorCategory.POSITION_CONSISTENCY: self._compensate_position_consistency,
            PortfolioErrorCategory.CALCULATION_ERROR: self._compensate_calculation_error,
            PortfolioErrorCategory.VALIDATION_ERROR: self._compensate_validation_error,
            PortfolioErrorCategory.EXTERNAL_SERVICE: self._compensate_external_service_error,
            PortfolioErrorCategory.CONCURRENCY_ISSUE: self._compensate_concurrency_issue
        }
    
    async def handle_portfolio_error(
        self, 
        exception: Exception, 
        context: Dict[str, Any],
        operation: str,
        trading_pair: Optional[str] = None
    ) -> Any:
        """
        Handle portfolio error with comprehensive classification and recovery.
        
        This replaces the basic "raise # Re-raise for now" approach with
        sophisticated error handling including classification, compensation,
        and recovery strategies.
        """
        try:
            # Classify the error
            portfolio_error = await self._classify_error(
                exception, context, operation, trading_pair
            )
            
            # Log error with comprehensive context
            await self._log_portfolio_error(portfolio_error)
            
            # Add to error history
            self.error_history.append(portfolio_error)
            
            # Execute recovery strategy
            return await self._execute_recovery_strategy(portfolio_error)
            
        except Exception as recovery_error:
            self.logger.error(
                f"Portfolio error recovery failed: {recovery_error}",
                exc_info=True,
                extra={
                    "original_error": str(exception),
                    "recovery_error": str(recovery_error),
                    "operation": operation,
                    "trading_pair": trading_pair
                }
            )
            # Last resort: escalate original error
            raise exception from recovery_error
    
    async def _classify_error(
        self, 
        exception: Exception, 
        context: Dict[str, Any],
        operation: str,
        trading_pair: Optional[str]
    ) -> PortfolioError:
        """Classify error into appropriate category with recovery strategy."""
        error_id = str(uuid.uuid4())
        
        # Classify based on exception type and context
        if isinstance(exception, (ConnectionError, TimeoutError)):
            category = PortfolioErrorCategory.DATABASE_ERROR
            severity = ErrorSeverity.HIGH
            recovery_strategy = RecoveryStrategy.DELAYED_RETRY
            
        elif isinstance(exception, ValueError) and "position" in str(exception).lower():
            category = PortfolioErrorCategory.POSITION_CONSISTENCY
            severity = ErrorSeverity.CRITICAL
            recovery_strategy = RecoveryStrategy.COMPENSATE
            
        elif isinstance(exception, (ArithmeticError, OverflowError)):
            category = PortfolioErrorCategory.CALCULATION_ERROR
            severity = ErrorSeverity.MEDIUM
            recovery_strategy = RecoveryStrategy.COMPENSATE
            
        elif isinstance(exception, (TypeError, AttributeError)):
            category = PortfolioErrorCategory.VALIDATION_ERROR
            severity = ErrorSeverity.MEDIUM
            recovery_strategy = RecoveryStrategy.IMMEDIATE_RETRY
            
        elif "external" in str(exception).lower() or "api" in str(exception).lower():
            category = PortfolioErrorCategory.EXTERNAL_SERVICE
            severity = ErrorSeverity.HIGH
            recovery_strategy = RecoveryStrategy.GRACEFUL_DEGRADATION
            
        elif "lock" in str(exception).lower() or "concurrent" in str(exception).lower():
            category = PortfolioErrorCategory.CONCURRENCY_ISSUE
            severity = ErrorSeverity.MEDIUM
            recovery_strategy = RecoveryStrategy.DELAYED_RETRY
            
        else:
            category = PortfolioErrorCategory.CONFIGURATION_ERROR
            severity = ErrorSeverity.MEDIUM
            recovery_strategy = RecoveryStrategy.ESCALATE
        
        return PortfolioError(
            error_id=error_id,
            category=category,
            severity=severity,
            recovery_strategy=recovery_strategy,
            original_exception=exception,
            context=context,
            timestamp=datetime.now(timezone.utc),
            trading_pair=trading_pair,
            operation=operation
        )
    
    async def _execute_recovery_strategy(self, portfolio_error: PortfolioError) -> Any:
        """Execute the determined recovery strategy."""
        strategy_map = {
            RecoveryStrategy.IMMEDIATE_RETRY: self._immediate_retry,
            RecoveryStrategy.DELAYED_RETRY: self._delayed_retry,
            RecoveryStrategy.COMPENSATE: self._compensate,
            RecoveryStrategy.ISOLATE: self._isolate,
            RecoveryStrategy.ESCALATE: self._escalate,
            RecoveryStrategy.GRACEFUL_DEGRADATION: self._graceful_degradation
        }
        
        strategy_func = strategy_map.get(portfolio_error.recovery_strategy, self._escalate)
        return await strategy_func(portfolio_error)
    
    async def _compensate(self, portfolio_error: PortfolioError) -> Any:
        """Execute compensation strategy based on error category."""
        compensation_handler = self.compensation_handlers.get(portfolio_error.category)
        
        if compensation_handler:
            try:
                return await compensation_handler(portfolio_error)
            except Exception as comp_error:
                self.logger.error(
                    f"Compensation failed: {comp_error}",
                    extra={"portfolio_error_id": portfolio_error.error_id}
                )
                # Escalate if compensation fails
                return await self._escalate(portfolio_error)
        
        # No compensation handler available
        return await self._escalate(portfolio_error)
    
    async def _compensate_database_error(self, portfolio_error: PortfolioError) -> Any:
        """Compensate for database errors."""
        self.logger.info(f"Compensating database error: {portfolio_error.error_id}")
        
        # Try alternative database connection or read-only mode
        try:
            # Implement database failover logic
            # Return cached data or default values
            return await self._get_cached_data(portfolio_error.context)
        except Exception:
            # If compensation fails, use graceful degradation
            return await self._graceful_degradation(portfolio_error)
    
    async def _compensate_position_consistency(self, portfolio_error: PortfolioError) -> Any:
        """Compensate for position consistency errors."""
        self.logger.warning(f"Compensating position consistency error: {portfolio_error.error_id}")
        
        try:
            # Trigger position reconciliation
            await self._trigger_position_reconciliation(portfolio_error.trading_pair)
            
            # Return safe default or recalculated position
            return await self._recalculate_position_safely(portfolio_error.context)
        except Exception:
            return await self._escalate(portfolio_error)


class EnhancedPositionManager:
    """
    Enhanced position manager with comprehensive error handling.
    
    This replaces basic error handling with sophisticated error classification,
    recovery mechanisms, and legacy code management.
    """
    
    def __init__(self, position_repository, logger: LoggerService):
        self.position_repository = position_repository
        self.logger = logger
        self.error_handler = PortfolioErrorHandler(logger)
        self.legacy_compatibility_layer = LegacyCompatibilityLayer(logger)
        self._source_module = "EnhancedPositionManager"
    
    async def create_position_with_enhanced_error_handling(
        self, 
        trading_pair: str, 
        initial_data: Dict[str, Any]
    ) -> Any:
        """
        Create position with comprehensive error handling.
        
        This replaces the basic "raise # Re-raise for now" approach with
        sophisticated error handling and recovery mechanisms.
        """
        operation_context = {
            "operation": "create_position",
            "trading_pair": trading_pair,
            "initial_data_keys": list(initial_data.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Validate input data
            await self._validate_position_data(trading_pair, initial_data)
            
            # Check for existing position conflicts
            await self._check_position_conflicts(trading_pair)
            
            # Create position with transaction safety
            position = await self._create_position_safely(trading_pair, initial_data)
            
            self.logger.info(
                f"Position created successfully for {trading_pair}",
                extra=operation_context
            )
            
            return position
            
        except Exception as e:
            # Use sophisticated error handling instead of basic re-raise
            return await self.error_handler.handle_portfolio_error(
                exception=e,
                context=operation_context,
                operation="create_position",
                trading_pair=trading_pair
            )
    
    async def get_total_realized_pnl_with_legacy_support(
        self, 
        trading_pair: Optional[str] = None
    ) -> Decimal:
        """
        Get total realized PnL with proper legacy code management.
        
        This replaces the "For now, keeping if the internal logic still uses them"
        approach with proper legacy compatibility and migration planning.
        """
        operation_context = {
            "operation": "get_total_realized_pnl",
            "trading_pair": trading_pair,
            "legacy_mode": False
        }
        
        try:
            # Check if legacy compatibility is needed
            if await self.legacy_compatibility_layer.requires_legacy_support():
                self.logger.info("Using legacy compatibility layer for PnL calculation")
                operation_context["legacy_mode"] = True
                return await self._get_total_realized_pnl_legacy(trading_pair)
            
            # Use modern implementation
            return await self._get_total_realized_pnl_modern(trading_pair)
            
        except Exception as e:
            return await self.error_handler.handle_portfolio_error(
                exception=e,
                context=operation_context,
                operation="get_total_realized_pnl",
                trading_pair=trading_pair
            )
    
    async def _get_total_realized_pnl_modern(self, trading_pair: Optional[str] = None) -> Decimal:
        """Modern implementation of total realized PnL calculation."""
        if trading_pair:
            position = await self.position_repository.get_position_by_pair(trading_pair)
            return position.realized_pnl if position and position.realized_pnl else Decimal(0)
        
        # Efficient aggregation query for all pairs
        return await self.position_repository.get_total_realized_pnl()
    
    async def _get_total_realized_pnl_legacy(self, trading_pair: Optional[str] = None) -> Decimal:
        """Legacy implementation with compatibility layer."""
        # This handles the "For now, keeping if the internal logic still uses them" scenario
        return await self.legacy_compatibility_layer.calculate_legacy_pnl(trading_pair)


class LegacyCompatibilityLayer:
    """
    Legacy code compatibility layer with migration planning.
    
    This provides structured approach to legacy code management replacing
    uncertain "for now" approaches with clear migration strategies.
    """
    
    def __init__(self, logger: LoggerService):
        self.logger = logger
        self.migration_tracker = LegacyMigrationTracker()
    
    async def requires_legacy_support(self) -> bool:
        """Determine if legacy support is still required."""
        # Check system configuration and usage patterns
        legacy_usage = await self._check_legacy_usage_patterns()
        migration_status = await self.migration_tracker.get_migration_status()
        
        return legacy_usage["active_legacy_calls"] > 0 or not migration_status["migration_complete"]
    
    async def calculate_legacy_pnl(self, trading_pair: Optional[str] = None) -> Decimal:
        """Legacy PnL calculation with deprecation warnings."""
        self.logger.warning(
            "Using legacy PnL calculation - scheduled for deprecation",
            extra={
                "deprecation_date": "2024-12-31",
                "migration_ticket": "TECH-1234",
                "trading_pair": trading_pair
            }
        )
        
        # Track legacy usage for migration planning
        await self.migration_tracker.track_legacy_usage("calculate_legacy_pnl", trading_pair)
        
        # Legacy implementation (simplified for example)
        # This would contain the actual legacy logic
        return Decimal("0.0")


class LegacyMigrationTracker:
    """Track legacy code usage and migration progress."""
    
    def __init__(self):
        self.usage_stats: Dict[str, int] = {}
        self.migration_deadlines: Dict[str, datetime] = {}
    
    async def track_legacy_usage(self, function_name: str, context: Any) -> None:
        """Track usage of legacy functions for migration planning."""
        self.usage_stats[function_name] = self.usage_stats.get(function_name, 0) + 1
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status and recommendations."""
        return {
            "migration_complete": len(self.usage_stats) == 0,
            "active_legacy_functions": list(self.usage_stats.keys()),
            "usage_counts": self.usage_stats,
            "migration_deadlines": self.migration_deadlines
        }
```

## Testing Strategy

1. **Schema Management Testing**
   - Version detection accuracy
   - Migration safety validation
   - Rollback capability testing
   - Performance impact assessment

2. **Exception Handler Testing**
   - Configuration validation
   - Strategy effectiveness
   - Performance under load
   - Context awareness accuracy

3. **Portfolio Error Handling Testing**
   - Error classification accuracy
   - Recovery strategy effectiveness
   - Compensation mechanism validation
   - State consistency guarantees

4. **Legacy Code Management Testing**
   - Compatibility layer functionality
   - Migration tracking accuracy
   - Deprecation workflow validation
   - Performance comparison testing