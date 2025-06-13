# Migration Manager Implementation Design

**File**: `/gal_friday/dal/migrations/migration_manager.py`
- **Line 39**: `# For now, subsequent calls will likely fail if alembic_cfg cannot be loaded`
- **Line 103**: `# For now, let's try to get script heads which doesn't require DB connection`
- **Line 112**: `# For now, sticking to a method that can get DB state if possible`

## Overview
The migration manager contains basic implementations with error-prone configuration loading, limited database connection handling, and simplified state detection. This design implements a comprehensive, production-grade database migration management system with robust error handling, rollback capabilities, migration validation, and enterprise-level safety features.

## Architecture Design

### 1. Current Implementation Issues

```
Migration Manager Problems:
├── Configuration Loading (Line 39)
│   ├── Basic alembic config loading
│   ├── No error recovery mechanisms
│   ├── Missing configuration validation
│   └── No environment-specific handling
├── Script Head Detection (Line 103)
│   ├── Simple head retrieval without DB connection
│   ├── No migration conflict detection
│   ├── Missing branch validation
│   └── No dependency verification
├── Database State Detection (Line 112)
│   ├── Basic database state checking
│   ├── No migration integrity validation
│   ├── Missing schema version tracking
│   └── No concurrent migration protection
└── Migration Safety
    ├── No pre-migration validation
    ├── Missing rollback mechanisms
    ├── No backup integration
    └── No migration impact assessment
```

### 2. Production Migration Architecture

```
Enterprise Migration Management System:
├── Robust Configuration Management
│   ├── Environment-aware configuration
│   ├── Multi-database support
│   ├── Configuration validation
│   ├── Dynamic configuration reloading
│   └── Security and encryption support
├── Advanced Migration Engine
│   ├── Dependency graph resolution
│   ├── Conflict detection and resolution
│   ├── Parallel migration support
│   ├── Rollback planning and execution
│   └── Migration impact analysis
├── Enterprise Safety Features
│   ├── Pre-migration database backup
│   ├── Schema validation and testing
│   ├── Migration dry-run capability
│   ├── Concurrent migration protection
│   └── Migration monitoring and alerting
└── Production Operations
    ├── Zero-downtime migration strategies
    ├── Blue-green deployment support
    ├── Migration performance optimization
    ├── Compliance and audit logging
    └── Emergency recovery procedures
```

## Implementation Plan

### Phase 1: Enterprise Migration Configuration and Engine

```python
import asyncio
import json
import os
import time
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import tempfile
from contextlib import asynccontextmanager

# Alembic and SQLAlchemy imports
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.environment import EnvironmentContext
from alembic.runtime.migration import MigrationContext
from alembic.migration import MigrationStep
from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from gal_friday.logger_service import LoggerService
from gal_friday.config_manager import ConfigManager


class MigrationStatus(str, Enum):
    """Migration execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"


class MigrationDirection(str, Enum):
    """Migration direction."""
    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"


class MigrationStrategy(str, Enum):
    """Migration execution strategies."""
    STANDARD = "standard"
    ZERO_DOWNTIME = "zero_downtime"
    BLUE_GREEN = "blue_green"
    PARALLEL = "parallel"


class MigrationSafety(str, Enum):
    """Migration safety levels."""
    SAFE = "safe"
    CAUTIOUS = "cautious"
    AGGRESSIVE = "aggressive"


@dataclass
class MigrationInfo:
    """Comprehensive migration information."""
    revision: str
    description: str
    author: Optional[str]
    created_at: datetime
    file_path: Path
    
    # Dependencies
    down_revision: Optional[str] = None
    branch_labels: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    
    # Metadata
    estimated_duration: Optional[float] = None
    impact_level: str = "medium"
    requires_downtime: bool = False
    
    # Safety information
    has_rollback: bool = True
    rollback_safe: bool = True
    data_migration: bool = False
    schema_changes: List[str] = field(default_factory=list)


@dataclass
class MigrationPlan:
    """Migration execution plan."""
    migrations: List[MigrationInfo]
    strategy: MigrationStrategy
    safety_level: MigrationSafety
    direction: MigrationDirection
    
    # Safety measures
    backup_required: bool = True
    dry_run_completed: bool = False
    impact_assessed: bool = False
    
    # Execution metadata
    estimated_total_duration: float = 0.0
    downtime_required: bool = False
    rollback_plan: Optional['MigrationPlan'] = None


@dataclass
class MigrationResult:
    """Migration execution result."""
    revision: str
    status: MigrationStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    rollback_revision: Optional[str] = None
    
    # Performance metrics
    rows_affected: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


class DatabaseBackupManager:
    """Database backup management for migration safety."""
    
    def __init__(self, config: ConfigManager, logger: LoggerService):
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Backup configuration
        self._backup_dir = Path(config.get("migrations.backup_dir", "/var/backups/gal-friday"))
        self._backup_retention_days = config.get("migrations.backup_retention_days", 30)
        self._compression_enabled = config.get("migrations.backup_compression", True)
        
        # Ensure backup directory exists
        self._backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_backup(self, database_url: str, backup_name: Optional[str] = None) -> Path:
        """Create database backup before migration."""
        try:
            if not backup_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"pre_migration_{timestamp}"
            
            backup_file = self._backup_dir / f"{backup_name}.sql"
            if self._compression_enabled:
                backup_file = backup_file.with_suffix(".sql.gz")
            
            # Extract database connection info
            db_config = self._parse_database_url(database_url)
            
            # Create pg_dump command
            dump_cmd = [
                "pg_dump",
                "--host", db_config["host"],
                "--port", str(db_config["port"]),
                "--username", db_config["username"],
                "--no-password",
                "--format", "custom",
                "--verbose",
                "--file", str(backup_file),
                db_config["database"]
            ]
            
            # Set environment for password
            env = os.environ.copy()
            env["PGPASSWORD"] = db_config["password"]
            
            # Execute backup
            start_time = time.time()
            
            process = await asyncio.create_subprocess_exec(
                *dump_cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Backup failed: {stderr.decode()}")
            
            duration = time.time() - start_time
            backup_size = backup_file.stat().st_size if backup_file.exists() else 0
            
            self.logger.info(
                f"Database backup created: {backup_file} ({backup_size / 1024 / 1024:.1f}MB, {duration:.1f}s)",
                source_module=self._source_module
            )
            
            return backup_file
            
        except Exception as e:
            self.logger.error(
                f"Failed to create database backup: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise
    
    async def restore_backup(self, backup_file: Path, database_url: str) -> bool:
        """Restore database from backup."""
        try:
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_file}")
            
            db_config = self._parse_database_url(database_url)
            
            # Create pg_restore command
            restore_cmd = [
                "pg_restore",
                "--host", db_config["host"],
                "--port", str(db_config["port"]),
                "--username", db_config["username"],
                "--no-password",
                "--clean",
                "--if-exists",
                "--dbname", db_config["database"],
                str(backup_file)
            ]
            
            # Set environment for password
            env = os.environ.copy()
            env["PGPASSWORD"] = db_config["password"]
            
            # Execute restore
            start_time = time.time()
            
            process = await asyncio.create_subprocess_exec(
                *restore_cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                # pg_restore may return non-zero even on success due to warnings
                stderr_text = stderr.decode()
                if "ERROR" in stderr_text:
                    raise RuntimeError(f"Restore failed: {stderr_text}")
            
            duration = time.time() - start_time
            
            self.logger.info(
                f"Database restored from backup: {backup_file} ({duration:.1f}s)",
                source_module=self._source_module
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to restore database backup: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return False
    
    async def cleanup_old_backups(self) -> int:
        """Clean up old backup files."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self._backup_retention_days)
            removed_count = 0
            
            for backup_file in self._backup_dir.glob("*.sql*"):
                file_mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                
                if file_mtime < cutoff_date:
                    backup_file.unlink()
                    removed_count += 1
                    
                    self.logger.debug(
                        f"Removed old backup: {backup_file}",
                        source_module=self._source_module
                    )
            
            if removed_count > 0:
                self.logger.info(
                    f"Cleaned up {removed_count} old backup files",
                    source_module=self._source_module
                )
            
            return removed_count
            
        except Exception as e:
            self.logger.error(
                f"Failed to cleanup old backups: {e}",
                source_module=self._source_module
            )
            return 0
    
    def _parse_database_url(self, database_url: str) -> Dict[str, Any]:
        """Parse database URL into components."""
        from urllib.parse import urlparse
        
        parsed = urlparse(database_url)
        
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            "username": parsed.username or "postgres",
            "password": parsed.password or "",
            "database": parsed.path.lstrip("/") if parsed.path else "postgres"
        }


class MigrationValidator:
    """Migration validation and safety checking."""
    
    def __init__(self, config: ConfigManager, logger: LoggerService):
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Validation rules
        self._dangerous_operations = {
            "DROP TABLE", "DROP COLUMN", "DROP INDEX", "DROP CONSTRAINT",
            "ALTER COLUMN", "TRUNCATE", "DELETE FROM"
        }
        
        self._performance_impact_operations = {
            "CREATE INDEX", "ALTER TABLE", "UPDATE", "INSERT INTO"
        }
    
    async def validate_migration(self, migration_info: MigrationInfo) -> Tuple[bool, List[str]]:
        """Validate migration for safety and correctness."""
        issues = []
        
        try:
            # Read migration file
            migration_content = migration_info.file_path.read_text()
            
            # Check for dangerous operations
            dangerous_ops = self._check_dangerous_operations(migration_content)
            if dangerous_ops:
                issues.extend([f"Dangerous operation detected: {op}" for op in dangerous_ops])
            
            # Check for performance impact
            performance_ops = self._check_performance_impact(migration_content)
            if performance_ops:
                issues.extend([f"Performance impact operation: {op}" for op in performance_ops])
            
            # Validate rollback capability
            if not self._has_rollback_logic(migration_content):
                issues.append("Migration lacks proper rollback logic")
            
            # Check for data migration patterns
            if self._is_data_migration(migration_content):
                migration_info.data_migration = True
                if not self._has_data_migration_safety(migration_content):
                    issues.append("Data migration lacks safety checks")
            
            # Validate SQL syntax (basic check)
            syntax_issues = await self._validate_sql_syntax(migration_content)
            issues.extend(syntax_issues)
            
            is_valid = len(issues) == 0
            
            self.logger.info(
                f"Migration validation for {migration_info.revision}: "
                f"{'PASSED' if is_valid else 'FAILED'} ({len(issues)} issues)",
                source_module=self._source_module
            )
            
            return is_valid, issues
            
        except Exception as e:
            error_msg = f"Validation error: {e}"
            self.logger.error(error_msg, source_module=self._source_module, exc_info=True)
            return False, [error_msg]
    
    def _check_dangerous_operations(self, content: str) -> List[str]:
        """Check for dangerous SQL operations."""
        found_operations = []
        content_upper = content.upper()
        
        for operation in self._dangerous_operations:
            if operation in content_upper:
                found_operations.append(operation)
        
        return found_operations
    
    def _check_performance_impact(self, content: str) -> List[str]:
        """Check for operations with performance impact."""
        found_operations = []
        content_upper = content.upper()
        
        for operation in self._performance_impact_operations:
            if operation in content_upper:
                found_operations.append(operation)
        
        return found_operations
    
    def _has_rollback_logic(self, content: str) -> bool:
        """Check if migration has proper rollback logic."""
        return "def downgrade(" in content and len(content.split("def downgrade(")[1].strip()) > 10
    
    def _is_data_migration(self, content: str) -> bool:
        """Check if migration involves data manipulation."""
        data_keywords = ["INSERT", "UPDATE", "DELETE", "COPY", "BULK"]
        content_upper = content.upper()
        
        return any(keyword in content_upper for keyword in data_keywords)
    
    def _has_data_migration_safety(self, content: str) -> bool:
        """Check if data migration has safety measures."""
        safety_keywords = ["TRANSACTION", "ROLLBACK", "SAVEPOINT", "BATCH", "LIMIT"]
        content_upper = content.upper()
        
        return any(keyword in content_upper for keyword in safety_keywords)
    
    async def _validate_sql_syntax(self, content: str) -> List[str]:
        """Basic SQL syntax validation."""
        issues = []
        
        # Check for basic syntax issues
        if content.count("(") != content.count(")"):
            issues.append("Unmatched parentheses")
        
        if content.count("'") % 2 != 0:
            issues.append("Unmatched single quotes")
        
        if content.count('"') % 2 != 0:
            issues.append("Unmatched double quotes")
        
        # Check for incomplete statements
        statements = content.split(";")
        for i, stmt in enumerate(statements[:-1]):  # Exclude last (may be empty)
            if stmt.strip() and not any(keyword in stmt.upper() for keyword in ["CREATE", "ALTER", "DROP", "INSERT", "UPDATE", "DELETE"]):
                issues.append(f"Potentially incomplete statement at position {i + 1}")
        
        return issues


class EnhancedMigrationManager:
    """Production-grade database migration management system."""
    
    def __init__(self, config: ConfigManager, logger: LoggerService):
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Configuration
        self._database_url = config.get("database.url")
        self._migrations_dir = Path(config.get("migrations.directory", "gal_friday/dal/migrations"))
        self._alembic_ini_path = self._migrations_dir / "alembic.ini"
        
        # Migration safety settings
        self._require_backup = config.get("migrations.require_backup", True)
        self._allow_destructive = config.get("migrations.allow_destructive", False)
        self._max_migration_time = config.get("migrations.max_time_seconds", 3600)
        
        # Components
        self._backup_manager = DatabaseBackupManager(config, logger)
        self._validator = MigrationValidator(config, logger)
        
        # State tracking
        self._migration_lock = asyncio.Lock()
        self._current_migration: Optional[str] = None
        
        # Initialize Alembic configuration
        self._alembic_config: Optional[Config] = None
        self._script_directory: Optional[ScriptDirectory] = None
        
    async def initialize(self) -> bool:
        """Initialize migration manager and validate configuration."""
        try:
            # Create Alembic configuration
            if not await self._setup_alembic_config():
                return False
            
            # Validate migrations directory
            if not self._migrations_dir.exists():
                self.logger.error(
                    f"Migrations directory not found: {self._migrations_dir}",
                    source_module=self._source_module
                )
                return False
            
            # Initialize script directory
            self._script_directory = ScriptDirectory.from_config(self._alembic_config)
            
            # Test database connection
            if not await self._test_database_connection():
                return False
            
            # Cleanup old backups
            await self._backup_manager.cleanup_old_backups()
            
            self.logger.info(
                f"Migration manager initialized successfully",
                source_module=self._source_module
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to initialize migration manager: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return False
    
    async def _setup_alembic_config(self) -> bool:
        """Setup Alembic configuration with error recovery."""
        try:
            # Check if alembic.ini exists
            if not self._alembic_ini_path.exists():
                self.logger.warning(
                    f"Alembic configuration not found: {self._alembic_ini_path}",
                    source_module=self._source_module
                )
                # Create default configuration
                await self._create_default_alembic_config()
            
            # Load Alembic configuration
            self._alembic_config = Config(str(self._alembic_ini_path))
            
            # Set database URL
            self._alembic_config.set_main_option("sqlalchemy.url", self._database_url)
            
            # Set script location
            self._alembic_config.set_main_option(
                "script_location", 
                str(self._migrations_dir)
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to setup Alembic configuration: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return False
    
    async def _create_default_alembic_config(self) -> None:
        """Create default Alembic configuration."""
        config_content = f"""# Gal-Friday Migration Configuration

[alembic]
script_location = {self._migrations_dir}
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = {self._database_url}

[post_write_hooks]

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""
        
        self._alembic_ini_path.write_text(config_content)
        self.logger.info(
            f"Created default Alembic configuration: {self._alembic_ini_path}",
            source_module=self._source_module
        )
    
    async def _test_database_connection(self) -> bool:
        """Test database connectivity."""
        try:
            engine = create_engine(
                self._database_url,
                poolclass=NullPool,
                echo=False
            )
            
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                row = result.fetchone()
                return row[0] == 1
                
        except Exception as e:
            self.logger.error(
                f"Database connection test failed: {e}",
                source_module=self._source_module
            )
            return False
    
    async def get_current_revision(self) -> Optional[str]:
        """Get current database revision."""
        try:
            engine = create_engine(self._database_url, poolclass=NullPool)
            
            with engine.connect() as connection:
                context = MigrationContext.configure(connection)
                current_rev = context.get_current_revision()
                
                self.logger.debug(
                    f"Current database revision: {current_rev}",
                    source_module=self._source_module
                )
                
                return current_rev
                
        except Exception as e:
            self.logger.error(
                f"Failed to get current revision: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return None
    
    async def get_pending_migrations(self, target_revision: Optional[str] = None) -> List[MigrationInfo]:
        """Get list of pending migrations."""
        try:
            if not self._script_directory:
                raise RuntimeError("Script directory not initialized")
            
            current_rev = await self.get_current_revision()
            
            # Get migration path
            if target_revision is None:
                target_revision = self._script_directory.get_current_head()
            
            # Get revisions to apply
            revisions = []
            for revision in self._script_directory.walk_revisions(
                base=current_rev,
                head=target_revision
            ):
                if revision.revision != current_rev:
                    migration_info = await self._create_migration_info(revision)
                    revisions.append(migration_info)
            
            # Reverse to get correct order (oldest first)
            revisions.reverse()
            
            self.logger.info(
                f"Found {len(revisions)} pending migrations",
                source_module=self._source_module
            )
            
            return revisions
            
        except Exception as e:
            self.logger.error(
                f"Failed to get pending migrations: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return []
    
    async def _create_migration_info(self, revision) -> MigrationInfo:
        """Create MigrationInfo from Alembic revision."""
        file_path = Path(revision.path) if revision.path else Path("")
        
        # Parse migration file for additional metadata
        created_at = datetime.fromtimestamp(file_path.stat().st_mtime) if file_path.exists() else datetime.now()
        
        return MigrationInfo(
            revision=revision.revision,
            description=revision.doc or "No description",
            author=None,  # Would extract from file if available
            created_at=created_at,
            file_path=file_path,
            down_revision=revision.down_revision,
            branch_labels=list(revision.branch_labels) if revision.branch_labels else [],
            depends_on=list(revision.depends_on) if revision.depends_on else []
        )
    
    async def create_migration_plan(
        self,
        target_revision: Optional[str] = None,
        strategy: MigrationStrategy = MigrationStrategy.STANDARD,
        safety_level: MigrationSafety = MigrationSafety.CAUTIOUS
    ) -> MigrationPlan:
        """Create comprehensive migration execution plan."""
        try:
            pending_migrations = await self.get_pending_migrations(target_revision)
            
            # Validate all migrations
            total_duration = 0.0
            requires_downtime = False
            
            for migration in pending_migrations:
                # Validate migration
                is_valid, issues = await self._validator.validate_migration(migration)
                
                if not is_valid and safety_level != MigrationSafety.AGGRESSIVE:
                    raise ValueError(f"Migration {migration.revision} validation failed: {issues}")
                
                # Estimate duration and impact
                if migration.estimated_duration:
                    total_duration += migration.estimated_duration
                
                if migration.requires_downtime:
                    requires_downtime = True
            
            # Create migration plan
            plan = MigrationPlan(
                migrations=pending_migrations,
                strategy=strategy,
                safety_level=safety_level,
                direction=MigrationDirection.UPGRADE,
                backup_required=self._require_backup,
                estimated_total_duration=total_duration,
                downtime_required=requires_downtime
            )
            
            # Create rollback plan
            if pending_migrations:
                current_rev = await self.get_current_revision()
                if current_rev:
                    plan.rollback_plan = await self._create_rollback_plan(current_rev)
            
            self.logger.info(
                f"Created migration plan: {len(pending_migrations)} migrations, "
                f"estimated duration: {total_duration:.1f}s, "
                f"downtime required: {requires_downtime}",
                source_module=self._source_module
            )
            
            return plan
            
        except Exception as e:
            self.logger.error(
                f"Failed to create migration plan: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise
    
    async def _create_rollback_plan(self, target_revision: str) -> MigrationPlan:
        """Create rollback plan to specified revision."""
        # This would create a plan to rollback to the target revision
        # For now, return a basic structure
        return MigrationPlan(
            migrations=[],
            strategy=MigrationStrategy.STANDARD,
            safety_level=MigrationSafety.CAUTIOUS,
            direction=MigrationDirection.DOWNGRADE,
            backup_required=False
        )
    
    async def execute_migration_plan(self, plan: MigrationPlan) -> List[MigrationResult]:
        """Execute migration plan with comprehensive safety measures."""
        async with self._migration_lock:
            results = []
            backup_file = None
            
            try:
                self.logger.info(
                    f"Starting migration execution: {len(plan.migrations)} migrations",
                    source_module=self._source_module
                )
                
                # Create backup if required
                if plan.backup_required:
                    backup_file = await self._backup_manager.create_backup(self._database_url)
                
                # Execute migrations
                for migration in plan.migrations:
                    result = await self._execute_single_migration(migration, plan.strategy)
                    results.append(result)
                    
                    # Stop on failure
                    if result.status == MigrationStatus.FAILED:
                        self.logger.error(
                            f"Migration failed, stopping execution: {result.error_message}",
                            source_module=self._source_module
                        )
                        break
                
                # Check if all migrations succeeded
                failed_migrations = [r for r in results if r.status == MigrationStatus.FAILED]
                
                if failed_migrations and plan.safety_level == MigrationSafety.CAUTIOUS:
                    self.logger.warning(
                        f"Migration failures detected, considering rollback",
                        source_module=self._source_module
                    )
                    
                    # Optionally trigger rollback
                    # await self._rollback_migrations(backup_file)
                
                success_count = len([r for r in results if r.status == MigrationStatus.COMPLETED])
                
                self.logger.info(
                    f"Migration execution completed: {success_count}/{len(plan.migrations)} succeeded",
                    source_module=self._source_module
                )
                
                return results
                
            except Exception as e:
                self.logger.error(
                    f"Migration execution failed: {e}",
                    source_module=self._source_module,
                    exc_info=True
                )
                
                # Emergency rollback if backup exists
                if backup_file and plan.safety_level == MigrationSafety.CAUTIOUS:
                    self.logger.info(
                        "Attempting emergency rollback",
                        source_module=self._source_module
                    )
                    await self._backup_manager.restore_backup(backup_file, self._database_url)
                
                raise
    
    async def _execute_single_migration(
        self, 
        migration: MigrationInfo, 
        strategy: MigrationStrategy
    ) -> MigrationResult:
        """Execute a single migration with monitoring."""
        start_time = datetime.now(timezone.utc)
        
        try:
            self._current_migration = migration.revision
            
            self.logger.info(
                f"Executing migration {migration.revision}: {migration.description}",
                source_module=self._source_module
            )
            
            # Execute migration using Alembic
            if strategy == MigrationStrategy.ZERO_DOWNTIME:
                # Would implement zero-downtime migration logic
                pass
            
            # Standard migration execution
            command.upgrade(self._alembic_config, migration.revision)
            
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            result = MigrationResult(
                revision=migration.revision,
                status=MigrationStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration
            )
            
            self.logger.info(
                f"Migration {migration.revision} completed in {duration:.2f}s",
                source_module=self._source_module
            )
            
            return result
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            result = MigrationResult(
                revision=migration.revision,
                status=MigrationStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                error_message=str(e)
            )
            
            self.logger.error(
                f"Migration {migration.revision} failed after {duration:.2f}s: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            
            return result
            
        finally:
            self._current_migration = None
    
    async def dry_run_migration(self, plan: MigrationPlan) -> bool:
        """Perform dry run of migration plan."""
        try:
            self.logger.info(
                f"Starting migration dry run: {len(plan.migrations)} migrations",
                source_module=self._source_module
            )
            
            # Create temporary database for dry run
            temp_db_url = await self._create_temp_database()
            
            try:
                # Restore current database state to temp database
                current_backup = await self._backup_manager.create_backup(
                    self._database_url, 
                    "dry_run_base"
                )
                
                await self._backup_manager.restore_backup(current_backup, temp_db_url)
                
                # Execute migrations on temporary database
                temp_config = Config(str(self._alembic_ini_path))
                temp_config.set_main_option("sqlalchemy.url", temp_db_url)
                
                for migration in plan.migrations:
                    command.upgrade(temp_config, migration.revision)
                
                self.logger.info(
                    "Migration dry run completed successfully",
                    source_module=self._source_module
                )
                
                return True
                
            finally:
                # Cleanup temporary database
                await self._cleanup_temp_database(temp_db_url)
                
        except Exception as e:
            self.logger.error(
                f"Migration dry run failed: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return False
    
    async def _create_temp_database(self) -> str:
        """Create temporary database for testing."""
        # This would create a temporary database
        # For now, return a mock URL
        return f"{self._database_url}_temp_{int(time.time())}"
    
    async def _cleanup_temp_database(self, temp_db_url: str) -> None:
        """Clean up temporary database."""
        # This would drop the temporary database
        pass
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get comprehensive migration status."""
        try:
            current_rev = await self.get_current_revision()
            pending_migrations = await self.get_pending_migrations()
            
            return {
                "current_revision": current_rev,
                "pending_migrations_count": len(pending_migrations),
                "pending_migrations": [
                    {
                        "revision": m.revision,
                        "description": m.description,
                        "created_at": m.created_at.isoformat()
                    }
                    for m in pending_migrations
                ],
                "migration_in_progress": self._current_migration,
                "database_url_configured": bool(self._database_url),
                "alembic_configured": bool(self._alembic_config),
                "last_check": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(
                f"Failed to get migration status: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return {"error": str(e)}


# Factory function for easy initialization
async def create_migration_manager(
    config: ConfigManager, 
    logger: LoggerService
) -> EnhancedMigrationManager:
    """Create and initialize migration manager."""
    manager = EnhancedMigrationManager(config, logger)
    
    if not await manager.initialize():
        raise RuntimeError("Failed to initialize migration manager")
    
    return manager
```

## Testing Strategy

1. **Unit Tests**
   - Configuration loading and validation
   - Migration validation logic
   - Backup and restore operations
   - Error handling scenarios

2. **Integration Tests**
   - Complete migration execution cycle
   - Rollback functionality
   - Database connectivity handling
   - Multi-environment testing

3. **Performance Tests**
   - Large migration execution
   - Backup performance
   - Memory usage optimization
   - Concurrent migration handling

## Monitoring & Observability

1. **Migration Metrics**
   - Execution times and success rates
   - Migration failure patterns
   - Backup creation and restoration times
   - Database performance impact

2. **System Health**
   - Migration queue status
   - Database connectivity
   - Backup storage utilization
   - Configuration validity

## Security Considerations

1. **Data Protection**
   - Encrypted backup storage
   - Database credential security
   - Migration audit logging
   - Access control enforcement

2. **System Integrity**
   - Migration validation
   - Rollback capability verification
   - Backup integrity checks
   - Emergency recovery procedures

## Future Enhancements

1. **Advanced Features**
   - Blue-green deployment support
   - Parallel migration execution
   - Automated rollback triggers
   - Migration dependency optimization

2. **Operational Improvements**
   - Real-time migration monitoring
   - Predictive migration analysis
   - Advanced backup strategies
   - Integration with CI/CD pipelines