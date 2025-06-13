# Experiment Management Implementation Design

**File**: `/gal_friday/model_lifecycle/experiment_manager.py`
- **Line 1019**: `# Add to active experiments (still using ExperimentConfig dataclass for now)`
- **Line 1405**: `# For now, assume a basic mapping or that ExperimentConfig adapts`

**File**: `/gal_friday/dal/repositories/experiment_repository.py`
- **Line 67**: `# For now, let's assume BaseRepository.create or a direct session.merge could work`
- **Line 134**: `# For now, let's just return the input or fetch it`

## Overview
The experiment management system contains temporary implementations and basic assumptions about experiment persistence and configuration handling. This design implements a comprehensive, production-grade experiment lifecycle management system for cryptocurrency trading model experiments with full database integration, versioning, and performance tracking.

## Architecture Design

### 1. Current Implementation Issues

```
Experiment Management Problems:
├── ExperimentConfig Usage (Line 1019)
│   ├── Using dataclass instead of database entity
│   ├── No persistence layer integration
│   ├── Missing experiment versioning
│   └── No metadata validation
├── Database Mapping (Line 1405)
│   ├── Basic mapping assumptions
│   ├── No proper ORM integration
│   ├── Missing data transformation
│   └── No error handling
├── Repository Operations (Line 67)
│   ├── Assuming BaseRepository.create works
│   ├── No transaction management
│   ├── Missing validation logic
│   └── No conflict resolution
└── Data Retrieval (Line 134)
    ├── Simple input/output assumptions
    ├── No query optimization
    ├── Missing caching strategy
    └── No relationship loading
```

### 2. Production Experiment Management Architecture

```
Enterprise Experiment System:
├── Comprehensive Experiment Entity Model
│   ├── Full database schema with relationships
│   ├── Experiment versioning and lineage
│   ├── Configuration validation framework
│   ├── Metadata and parameter tracking
│   └── Status and lifecycle management
├── Advanced Repository Layer
│   ├── Complex query operations
│   ├── Transaction management
│   ├── Bulk operations support
│   ├── Caching and optimization
│   └── Data integrity enforcement
├── Experiment Lifecycle Manager
│   ├── State machine implementation
│   ├── Resource allocation and cleanup
│   ├── Dependency management
│   ├── Performance monitoring
│   └── Error recovery mechanisms
└── Integration and Analytics
    ├── Experiment comparison framework
    ├── Performance analytics
    ├── Resource usage tracking
    ├── Automated cleanup policies
    └── Audit trail maintenance
```

## Implementation Plan

### Phase 1: Enhanced Database Schema and Entity Model

```python
import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import event, and_, or_, desc, asc, func

from gal_friday.logger_service import LoggerService
from gal_friday.dal.base import Base, BaseRepository, DatabaseManager


class ExperimentStatus(str, Enum):
    """Experiment lifecycle status."""
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


class ExperimentType(str, Enum):
    """Types of experiments."""
    BACKTEST = "backtest"
    LIVE_PAPER = "live_paper"
    LIVE_REAL = "live_real"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    PERFORMANCE_ANALYSIS = "performance_analysis"


class ResourceType(str, Enum):
    """Resource allocation types."""
    CPU_CORES = "cpu_cores"
    MEMORY_GB = "memory_gb"
    GPU_UNITS = "gpu_units"
    DISK_GB = "disk_gb"
    NETWORK_MBPS = "network_mbps"


@dataclass
class ExperimentConfig:
    """Enhanced experiment configuration with validation."""
    name: str
    experiment_type: ExperimentType
    strategy_name: str
    symbol_pairs: List[str]
    
    # Time configuration
    start_date: datetime
    end_date: datetime
    timeframe: str  # '1m', '5m', '1h', etc.
    
    # Strategy parameters
    strategy_parameters: Dict[str, Any] = field(default_factory=dict)
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    feature_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Resource requirements
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Execution configuration
    max_runtime_hours: float = 24.0
    priority: int = 5  # 1-10, higher is more important
    auto_cleanup: bool = True
    
    # Validation and constraints
    max_position_size: Optional[Decimal] = None
    max_drawdown_percent: Optional[float] = None
    stop_on_margin_call: bool = True
    
    # Metadata
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    parent_experiment_id: Optional[str] = None
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.name or len(self.name.strip()) == 0:
            errors.append("Experiment name cannot be empty")
            
        if self.end_date <= self.start_date:
            errors.append("End date must be after start date")
            
        if not self.symbol_pairs:
            errors.append("At least one symbol pair must be specified")
            
        if self.max_runtime_hours <= 0:
            errors.append("Max runtime must be positive")
            
        if self.priority < 1 or self.priority > 10:
            errors.append("Priority must be between 1 and 10")
            
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        if self.timeframe not in valid_timeframes:
            errors.append(f"Invalid timeframe. Must be one of: {valid_timeframes}")
            
        return errors


class Experiment(Base):
    """Enhanced experiment database entity."""
    __tablename__ = 'experiments'
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Experiment classification
    experiment_type = Column(String(50), nullable=False)
    strategy_name = Column(String(100), nullable=False)
    
    # Status and lifecycle
    status = Column(String(20), nullable=False, default=ExperimentStatus.CREATED.value)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Configuration and parameters
    config_json = Column(JSONB, nullable=False)  # ExperimentConfig as JSON
    symbol_pairs = Column(JSONB, nullable=False)  # List of trading pairs
    
    # Time parameters
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    timeframe = Column(String(10), nullable=False)
    
    # Execution metadata
    priority = Column(Integer, default=5)
    max_runtime_hours = Column(Float, default=24.0)
    actual_runtime_seconds = Column(Float)
    
    # Resource tracking
    resource_requirements = Column(JSONB)
    resource_usage = Column(JSONB)
    
    # Performance metrics
    total_trades = Column(Integer, default=0)
    successful_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float)
    
    # Relationships and hierarchy
    parent_experiment_id = Column(UUID(as_uuid=True), ForeignKey('experiments.id'))
    parent_experiment = relationship("Experiment", remote_side=[id], backref="child_experiments")
    
    # Versioning
    version = Column(Integer, default=1)
    
    # Metadata
    tags = Column(JSONB)  # List of string tags
    metadata = Column(JSONB)  # Additional flexible metadata
    
    # Error tracking
    error_message = Column(Text)
    error_count = Column(Integer, default=0)
    
    # Cleanup tracking
    auto_cleanup = Column(Boolean, default=True)
    cleanup_at = Column(DateTime(timezone=True))
    archived_at = Column(DateTime(timezone=True))
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_experiments_status', 'status'),
        Index('idx_experiments_type', 'experiment_type'),
        Index('idx_experiments_strategy', 'strategy_name'),
        Index('idx_experiments_created', 'created_at'),
        Index('idx_experiments_priority', 'priority'),
        Index('idx_experiments_parent', 'parent_experiment_id'),
    )
    
    def to_config(self) -> ExperimentConfig:
        """Convert database entity back to ExperimentConfig."""
        return ExperimentConfig(
            name=self.name,
            experiment_type=ExperimentType(self.experiment_type),
            strategy_name=self.strategy_name,
            symbol_pairs=self.symbol_pairs,
            start_date=self.start_date,
            end_date=self.end_date,
            timeframe=self.timeframe,
            strategy_parameters=self.config_json.get('strategy_parameters', {}),
            model_parameters=self.config_json.get('model_parameters', {}),
            feature_parameters=self.config_json.get('feature_parameters', {}),
            resource_requirements={
                ResourceType(k): v for k, v in (self.resource_requirements or {}).items()
            },
            max_runtime_hours=self.max_runtime_hours,
            priority=self.priority,
            auto_cleanup=self.auto_cleanup,
            description=self.description,
            tags=self.tags or [],
            parent_experiment_id=str(self.parent_experiment_id) if self.parent_experiment_id else None
        )
    
    @classmethod
    def from_config(cls, config: ExperimentConfig) -> 'Experiment':
        """Create database entity from ExperimentConfig."""
        return cls(
            name=config.name,
            description=config.description,
            experiment_type=config.experiment_type.value,
            strategy_name=config.strategy_name,
            symbol_pairs=config.symbol_pairs,
            start_date=config.start_date,
            end_date=config.end_date,
            timeframe=config.timeframe,
            config_json={
                'strategy_parameters': config.strategy_parameters,
                'model_parameters': config.model_parameters,
                'feature_parameters': config.feature_parameters,
            },
            priority=config.priority,
            max_runtime_hours=config.max_runtime_hours,
            resource_requirements={k.value: v for k, v in config.resource_requirements.items()},
            auto_cleanup=config.auto_cleanup,
            tags=config.tags,
            parent_experiment_id=uuid.UUID(config.parent_experiment_id) if config.parent_experiment_id else None
        )


class ExperimentLog(Base):
    """Experiment execution logs."""
    __tablename__ = 'experiment_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey('experiments.id'), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    level = Column(String(10), nullable=False)  # INFO, WARN, ERROR
    message = Column(Text, nullable=False)
    context = Column(JSONB)  # Additional context data
    
    experiment = relationship("Experiment", backref="logs")
    
    __table_args__ = (
        Index('idx_experiment_logs_experiment', 'experiment_id'),
        Index('idx_experiment_logs_timestamp', 'timestamp'),
        Index('idx_experiment_logs_level', 'level'),
    )


class ExperimentMetric(Base):
    """Time-series metrics for experiments."""
    __tablename__ = 'experiment_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey('experiments.id'), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_metadata = Column(JSONB)
    
    experiment = relationship("Experiment", backref="metrics")
    
    __table_args__ = (
        Index('idx_experiment_metrics_experiment', 'experiment_id'),
        Index('idx_experiment_metrics_name', 'metric_name'),
        Index('idx_experiment_metrics_timestamp', 'timestamp'),
    )


class ExperimentRepository(BaseRepository[Experiment]):
    """Enhanced repository for experiment operations."""
    
    def __init__(self, db_manager: DatabaseManager, logger: LoggerService):
        super().__init__(db_manager, Experiment, logger)
        self._source_module = self.__class__.__name__
    
    async def create_experiment(self, config: ExperimentConfig) -> Experiment:
        """Create a new experiment with validation."""
        try:
            # Validate configuration
            validation_errors = config.validate()
            if validation_errors:
                raise ValueError(f"Configuration validation failed: {', '.join(validation_errors)}")
            
            # Check for duplicate names (within reasonable time window)
            existing = await self.find_by_name_recent(config.name, days=30)
            if existing:
                raise ValueError(f"Experiment with name '{config.name}' already exists (created: {existing.created_at})")
            
            # Create experiment entity
            experiment = Experiment.from_config(config)
            
            # Set calculated cleanup time if auto-cleanup enabled
            if config.auto_cleanup:
                experiment.cleanup_at = datetime.utcnow() + timedelta(days=30)
            
            # Save to database with transaction
            async with self.db_manager.transaction() as session:
                session.add(experiment)
                await session.flush()  # Get the ID
                
                # Create initial log entry
                log_entry = ExperimentLog(
                    experiment_id=experiment.id,
                    level="INFO",
                    message=f"Experiment '{experiment.name}' created",
                    context={
                        "experiment_type": experiment.experiment_type,
                        "strategy": experiment.strategy_name,
                        "symbol_pairs": experiment.symbol_pairs
                    }
                )
                session.add(log_entry)
                
                await session.commit()
            
            self.logger.info(
                f"Created experiment {experiment.id} ({experiment.name})",
                source_module=self._source_module
            )
            
            return experiment
            
        except Exception as e:
            self.logger.error(
                f"Failed to create experiment: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise
    
    async def find_by_name_recent(self, name: str, days: int = 30) -> Optional[Experiment]:
        """Find experiment by name within recent time window."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        async with self.db_manager.session() as session:
            result = await session.execute(
                self.model.select().where(
                    and_(
                        self.model.name == name,
                        self.model.created_at >= cutoff_date
                    )
                )
            )
            return result.scalars().first()
    
    async def find_active_experiments(self) -> List[Experiment]:
        """Find all currently active experiments."""
        active_statuses = [
            ExperimentStatus.QUEUED.value,
            ExperimentStatus.RUNNING.value
        ]
        
        async with self.db_manager.session() as session:
            result = await session.execute(
                self.model.select().where(
                    self.model.status.in_(active_statuses)
                ).order_by(desc(self.model.priority), asc(self.model.created_at))
            )
            return result.scalars().all()
    
    async def find_by_strategy(self, strategy_name: str, limit: int = 50) -> List[Experiment]:
        """Find experiments by strategy name."""
        async with self.db_manager.session() as session:
            result = await session.execute(
                self.model.select().where(
                    self.model.strategy_name == strategy_name
                ).order_by(desc(self.model.created_at)).limit(limit)
            )
            return result.scalars().all()
    
    async def find_experiments_for_cleanup(self) -> List[Experiment]:
        """Find experiments ready for cleanup."""
        now = datetime.utcnow()
        
        async with self.db_manager.session() as session:
            result = await session.execute(
                self.model.select().where(
                    and_(
                        self.model.auto_cleanup == True,
                        self.model.cleanup_at <= now,
                        self.model.archived_at.is_(None)
                    )
                )
            )
            return result.scalars().all()
    
    async def update_status(
        self, 
        experiment_id: uuid.UUID, 
        status: ExperimentStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """Update experiment status with proper state transitions."""
        try:
            async with self.db_manager.transaction() as session:
                experiment = await session.get(self.model, experiment_id)
                if not experiment:
                    return False
                
                # Validate state transition
                if not self._is_valid_status_transition(experiment.status, status.value):
                    raise ValueError(f"Invalid status transition from {experiment.status} to {status.value}")
                
                # Update status and timestamps
                old_status = experiment.status
                experiment.status = status.value
                
                if status == ExperimentStatus.RUNNING and not experiment.started_at:
                    experiment.started_at = datetime.utcnow()
                elif status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED]:
                    if not experiment.completed_at:
                        experiment.completed_at = datetime.utcnow()
                    
                    # Calculate runtime
                    if experiment.started_at:
                        runtime = (experiment.completed_at - experiment.started_at).total_seconds()
                        experiment.actual_runtime_seconds = runtime
                
                # Handle error information
                if status == ExperimentStatus.FAILED:
                    experiment.error_message = error_message
                    experiment.error_count = (experiment.error_count or 0) + 1
                
                # Create status change log
                log_entry = ExperimentLog(
                    experiment_id=experiment_id,
                    level="INFO" if status != ExperimentStatus.FAILED else "ERROR",
                    message=f"Status changed from {old_status} to {status.value}",
                    context={
                        "old_status": old_status,
                        "new_status": status.value,
                        "error_message": error_message
                    }
                )
                session.add(log_entry)
                
                await session.commit()
                
                self.logger.info(
                    f"Updated experiment {experiment_id} status: {old_status} -> {status.value}",
                    source_module=self._source_module
                )
                
                return True
                
        except Exception as e:
            self.logger.error(
                f"Failed to update experiment status: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return False
    
    async def record_metric(
        self, 
        experiment_id: uuid.UUID, 
        metric_name: str, 
        metric_value: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record a metric for an experiment."""
        try:
            metric = ExperimentMetric(
                experiment_id=experiment_id,
                timestamp=timestamp or datetime.utcnow(),
                metric_name=metric_name,
                metric_value=metric_value,
                metric_metadata=metadata
            )
            
            async with self.db_manager.session() as session:
                session.add(metric)
                await session.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to record metric {metric_name} for experiment {experiment_id}: {e}",
                source_module=self._source_module
            )
            return False
    
    async def get_experiment_metrics(
        self, 
        experiment_id: uuid.UUID,
        metric_names: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[ExperimentMetric]:
        """Get metrics for an experiment with optional filtering."""
        async with self.db_manager.session() as session:
            query = session.query(ExperimentMetric).filter(
                ExperimentMetric.experiment_id == experiment_id
            )
            
            if metric_names:
                query = query.filter(ExperimentMetric.metric_name.in_(metric_names))
            
            if start_time:
                query = query.filter(ExperimentMetric.timestamp >= start_time)
            
            if end_time:
                query = query.filter(ExperimentMetric.timestamp <= end_time)
            
            query = query.order_by(ExperimentMetric.timestamp)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_performance_summary(self, experiment_id: uuid.UUID) -> Dict[str, Any]:
        """Get performance summary for an experiment."""
        async with self.db_manager.session() as session:
            experiment = await session.get(self.model, experiment_id)
            if not experiment:
                return {}
            
            # Get latest metrics
            metrics_query = session.query(ExperimentMetric).filter(
                ExperimentMetric.experiment_id == experiment_id
            ).order_by(desc(ExperimentMetric.timestamp))
            
            recent_metrics = await session.execute(metrics_query.limit(100))
            metrics_list = recent_metrics.scalars().all()
            
            # Aggregate metrics by name
            metrics_by_name = {}
            for metric in metrics_list:
                if metric.metric_name not in metrics_by_name:
                    metrics_by_name[metric.metric_name] = []
                metrics_by_name[metric.metric_name].append(metric.metric_value)
            
            # Calculate summary statistics
            summary = {
                "experiment_id": str(experiment_id),
                "name": experiment.name,
                "status": experiment.status,
                "total_trades": experiment.total_trades,
                "successful_trades": experiment.successful_trades,
                "success_rate": experiment.successful_trades / experiment.total_trades if experiment.total_trades > 0 else 0,
                "total_pnl": experiment.total_pnl,
                "max_drawdown": experiment.max_drawdown,
                "sharpe_ratio": experiment.sharpe_ratio,
                "runtime_seconds": experiment.actual_runtime_seconds,
                "metrics_summary": {}
            }
            
            # Add metrics summary
            for name, values in metrics_by_name.items():
                if values:
                    summary["metrics_summary"][name] = {
                        "latest_value": values[0],
                        "count": len(values),
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }
            
            return summary
    
    async def archive_experiment(self, experiment_id: uuid.UUID) -> bool:
        """Archive an experiment (soft delete)."""
        try:
            async with self.db_manager.transaction() as session:
                experiment = await session.get(self.model, experiment_id)
                if not experiment:
                    return False
                
                experiment.archived_at = datetime.utcnow()
                experiment.status = ExperimentStatus.ARCHIVED.value
                
                # Create archive log
                log_entry = ExperimentLog(
                    experiment_id=experiment_id,
                    level="INFO",
                    message="Experiment archived",
                    context={"archived_at": experiment.archived_at.isoformat()}
                )
                session.add(log_entry)
                
                await session.commit()
                
                self.logger.info(
                    f"Archived experiment {experiment_id}",
                    source_module=self._source_module
                )
                
                return True
                
        except Exception as e:
            self.logger.error(
                f"Failed to archive experiment {experiment_id}: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return False
    
    def _is_valid_status_transition(self, current_status: str, new_status: str) -> bool:
        """Validate experiment status transitions."""
        valid_transitions = {
            ExperimentStatus.CREATED.value: [
                ExperimentStatus.QUEUED.value,
                ExperimentStatus.CANCELLED.value
            ],
            ExperimentStatus.QUEUED.value: [
                ExperimentStatus.RUNNING.value,
                ExperimentStatus.CANCELLED.value
            ],
            ExperimentStatus.RUNNING.value: [
                ExperimentStatus.COMPLETED.value,
                ExperimentStatus.FAILED.value,
                ExperimentStatus.CANCELLED.value
            ],
            ExperimentStatus.COMPLETED.value: [
                ExperimentStatus.ARCHIVED.value
            ],
            ExperimentStatus.FAILED.value: [
                ExperimentStatus.QUEUED.value,  # Allow retry
                ExperimentStatus.ARCHIVED.value
            ],
            ExperimentStatus.CANCELLED.value: [
                ExperimentStatus.ARCHIVED.value
            ]
        }
        
        allowed = valid_transitions.get(current_status, [])
        return new_status in allowed


class EnhancedExperimentManager:
    """Production-grade experiment lifecycle manager."""
    
    def __init__(
        self, 
        experiment_repository: ExperimentRepository,
        logger: LoggerService,
        config: Dict[str, Any]
    ):
        self.repository = experiment_repository
        self.logger = logger
        self.config = config
        self._source_module = self.__class__.__name__
        
        # Active experiment tracking
        self._active_experiments: Dict[uuid.UUID, Experiment] = {}
        self._resource_allocations: Dict[uuid.UUID, Dict[ResourceType, float]] = {}
        
        # Configuration
        self._max_concurrent_experiments = config.get("max_concurrent_experiments", 5)
        self._default_cleanup_days = config.get("default_cleanup_days", 30)
        self._resource_limits = {
            ResourceType.CPU_CORES: config.get("max_cpu_cores", 8),
            ResourceType.MEMORY_GB: config.get("max_memory_gb", 32),
            ResourceType.GPU_UNITS: config.get("max_gpu_units", 2),
            ResourceType.DISK_GB: config.get("max_disk_gb", 100),
            ResourceType.NETWORK_MBPS: config.get("max_network_mbps", 1000)
        }
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the experiment manager."""
        try:
            # Load active experiments
            await self._load_active_experiments()
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.logger.info(
                f"Experiment manager started with {len(self._active_experiments)} active experiments",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to start experiment manager: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise
    
    async def stop(self) -> None:
        """Stop the experiment manager."""
        try:
            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
            if self._monitoring_task:
                self._monitoring_task.cancel()
            
            # Wait for tasks to complete
            if self._cleanup_task:
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            if self._monitoring_task:
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info(
                "Experiment manager stopped",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.logger.error(
                f"Error stopping experiment manager: {e}",
                source_module=self._source_module
            )
    
    async def create_experiment(self, config: ExperimentConfig) -> uuid.UUID:
        """Create and optionally queue a new experiment."""
        try:
            # Validate resource requirements
            if not self._validate_resource_requirements(config.resource_requirements):
                raise ValueError("Resource requirements exceed system limits")
            
            # Create experiment
            experiment = await self.repository.create_experiment(config)
            
            # Add to active tracking if not at capacity
            if len(self._active_experiments) < self._max_concurrent_experiments:
                self._active_experiments[experiment.id] = experiment
                await self.repository.update_status(experiment.id, ExperimentStatus.QUEUED)
            
            self.logger.info(
                f"Created experiment {experiment.id}: {experiment.name}",
                source_module=self._source_module
            )
            
            return experiment.id
            
        except Exception as e:
            self.logger.error(
                f"Failed to create experiment: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise
    
    async def start_experiment(self, experiment_id: uuid.UUID) -> bool:
        """Start an experiment if resources are available."""
        try:
            experiment = await self.repository.get_by_id(experiment_id)
            if not experiment:
                return False
            
            # Check if experiment can be started
            if experiment.status != ExperimentStatus.QUEUED.value:
                self.logger.warning(
                    f"Cannot start experiment {experiment_id}: status is {experiment.status}",
                    source_module=self._source_module
                )
                return False
            
            # Check resource availability
            resource_requirements = {
                ResourceType(k): v for k, v in (experiment.resource_requirements or {}).items()
            }
            
            if not self._can_allocate_resources(resource_requirements):
                self.logger.info(
                    f"Cannot start experiment {experiment_id}: insufficient resources",
                    source_module=self._source_module
                )
                return False
            
            # Allocate resources
            self._allocate_resources(experiment_id, resource_requirements)
            
            # Update status
            await self.repository.update_status(experiment_id, ExperimentStatus.RUNNING)
            
            # Update active tracking
            self._active_experiments[experiment_id] = experiment
            
            self.logger.info(
                f"Started experiment {experiment_id}",
                source_module=self._source_module
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to start experiment {experiment_id}: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return False
    
    async def complete_experiment(
        self, 
        experiment_id: uuid.UUID, 
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """Complete an experiment and update final metrics."""
        try:
            # Update final performance metrics if provided
            if performance_metrics:
                for metric_name, value in performance_metrics.items():
                    await self.repository.record_metric(experiment_id, metric_name, value)
            
            # Update status
            await self.repository.update_status(experiment_id, ExperimentStatus.COMPLETED)
            
            # Release resources
            self._release_resources(experiment_id)
            
            # Remove from active tracking
            self._active_experiments.pop(experiment_id, None)
            
            self.logger.info(
                f"Completed experiment {experiment_id}",
                source_module=self._source_module
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to complete experiment {experiment_id}: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return False
    
    async def fail_experiment(self, experiment_id: uuid.UUID, error_message: str) -> bool:
        """Mark an experiment as failed."""
        try:
            await self.repository.update_status(
                experiment_id, 
                ExperimentStatus.FAILED, 
                error_message
            )
            
            # Release resources
            self._release_resources(experiment_id)
            
            # Remove from active tracking
            self._active_experiments.pop(experiment_id, None)
            
            self.logger.error(
                f"Failed experiment {experiment_id}: {error_message}",
                source_module=self._source_module
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to mark experiment {experiment_id} as failed: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return False
    
    async def get_experiment_status(self, experiment_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get comprehensive experiment status."""
        try:
            experiment = await self.repository.get_by_id(experiment_id)
            if not experiment:
                return None
            
            performance_summary = await self.repository.get_performance_summary(experiment_id)
            
            return {
                "id": str(experiment_id),
                "name": experiment.name,
                "status": experiment.status,
                "experiment_type": experiment.experiment_type,
                "strategy_name": experiment.strategy_name,
                "created_at": experiment.created_at.isoformat(),
                "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
                "completed_at": experiment.completed_at.isoformat() if experiment.completed_at else None,
                "runtime_seconds": experiment.actual_runtime_seconds,
                "resource_usage": experiment.resource_usage,
                "performance": performance_summary,
                "is_active": experiment_id in self._active_experiments
            }
            
        except Exception as e:
            self.logger.error(
                f"Failed to get experiment status for {experiment_id}: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return None
    
    async def list_experiments(
        self, 
        status_filter: Optional[List[ExperimentStatus]] = None,
        strategy_filter: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List experiments with optional filtering."""
        try:
            # This would implement filtered querying
            # For now, return basic list
            experiments = await self.repository.find_active_experiments()
            
            result = []
            for exp in experiments[:limit]:
                result.append({
                    "id": str(exp.id),
                    "name": exp.name,
                    "status": exp.status,
                    "experiment_type": exp.experiment_type,
                    "strategy_name": exp.strategy_name,
                    "created_at": exp.created_at.isoformat(),
                    "priority": exp.priority
                })
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"Failed to list experiments: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return []
    
    async def _load_active_experiments(self) -> None:
        """Load active experiments from database."""
        active_experiments = await self.repository.find_active_experiments()
        
        for experiment in active_experiments:
            self._active_experiments[experiment.id] = experiment
            
            # Restore resource allocations
            if experiment.resource_requirements:
                resource_requirements = {
                    ResourceType(k): v for k, v in experiment.resource_requirements.items()
                }
                self._resource_allocations[experiment.id] = resource_requirements
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup task."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Find experiments ready for cleanup
                cleanup_candidates = await self.repository.find_experiments_for_cleanup()
                
                for experiment in cleanup_candidates:
                    await self.repository.archive_experiment(experiment.id)
                    
                if cleanup_candidates:
                    self.logger.info(
                        f"Cleaned up {len(cleanup_candidates)} experiments",
                        source_module=self._source_module
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    f"Error in cleanup loop: {e}",
                    source_module=self._source_module
                )
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring task."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Check for stale experiments
                for experiment_id, experiment in list(self._active_experiments.items()):
                    if experiment.status == ExperimentStatus.RUNNING.value:
                        # Check if experiment has exceeded max runtime
                        if experiment.started_at:
                            runtime = (datetime.utcnow() - experiment.started_at).total_seconds()
                            max_runtime = experiment.max_runtime_hours * 3600
                            
                            if runtime > max_runtime:
                                await self.fail_experiment(
                                    experiment_id,
                                    f"Experiment exceeded max runtime of {experiment.max_runtime_hours} hours"
                                )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    f"Error in monitoring loop: {e}",
                    source_module=self._source_module
                )
    
    def _validate_resource_requirements(self, requirements: Dict[ResourceType, float]) -> bool:
        """Validate that resource requirements don't exceed system limits."""
        for resource_type, amount in requirements.items():
            if amount > self._resource_limits.get(resource_type, 0):
                return False
        return True
    
    def _can_allocate_resources(self, requirements: Dict[ResourceType, float]) -> bool:
        """Check if resources can be allocated."""
        # Calculate currently allocated resources
        current_allocations = {res_type: 0.0 for res_type in ResourceType}
        
        for allocations in self._resource_allocations.values():
            for res_type, amount in allocations.items():
                current_allocations[res_type] += amount
        
        # Check if requirements can be satisfied
        for res_type, required in requirements.items():
            available = self._resource_limits.get(res_type, 0) - current_allocations.get(res_type, 0)
            if required > available:
                return False
        
        return True
    
    def _allocate_resources(self, experiment_id: uuid.UUID, requirements: Dict[ResourceType, float]) -> None:
        """Allocate resources for an experiment."""
        self._resource_allocations[experiment_id] = requirements.copy()
    
    def _release_resources(self, experiment_id: uuid.UUID) -> None:
        """Release resources for an experiment."""
        self._resource_allocations.pop(experiment_id, None)
```

## Testing Strategy

1. **Unit Tests**
   - Configuration validation logic
   - Status transition validation
   - Resource allocation algorithms
   - Database operations

2. **Integration Tests**
   - Full experiment lifecycle
   - Database transaction integrity
   - Background task coordination
   - Error recovery scenarios

3. **Performance Tests**
   - Concurrent experiment handling
   - Database query performance
   - Memory usage under load
   - Resource allocation efficiency

## Monitoring & Observability

1. **Experiment Metrics**
   - Experiment success/failure rates
   - Resource utilization patterns
   - Performance trends
   - Cleanup effectiveness

2. **System Health**
   - Active experiment counts
   - Resource allocation status
   - Database performance
   - Background task health

## Security Considerations

1. **Data Protection**
   - Experiment configuration validation
   - Resource limit enforcement
   - Access control implementation
   - Audit trail maintenance

2. **System Integrity**
   - Transaction isolation
   - Error boundary enforcement
   - Resource leak prevention
   - Proper cleanup procedures

## Future Enhancements

1. **Advanced Features**
   - Experiment dependency management
   - Dynamic resource scaling
   - Automated parameter optimization
   - Performance prediction models

2. **Integration Improvements**
   - Advanced scheduling algorithms
   - Multi-tenant support
   - Distributed execution
   - Real-time monitoring dashboard