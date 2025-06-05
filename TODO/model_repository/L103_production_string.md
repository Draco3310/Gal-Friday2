# Task: Replace "PRODUCTION" string with enum-based stage management

### 1. Context
- **File:** `gal_friday/dal/repositories/model_repository.py`
- **Line:** `103`
- **Keyword/Pattern:** `"Example"`
- **Current State:** Hardcoded "PRODUCTION" string that should be replaced with enum-based stage management

### 2. Problem Statement
The model repository uses hardcoded string literals for stage management, creating maintenance issues and potential bugs from string mismatches. This prevents proper stage validation and limits the system's ability to handle different deployment environments effectively.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Stage Enumeration:** Define comprehensive stage enum with all deployment environments
2. **Implement Stage Validation:** Validate stage transitions and permissions
3. **Build Stage Management System:** Complete lifecycle management for model stages
4. **Add Stage Configuration:** Environment-specific configuration for each stage
5. **Create Stage Monitoring:** Track model stage changes and transitions
6. **Build Audit Trail:** Log all stage changes for compliance and debugging

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import logging
import json

class ModelStage(str, Enum):
    """Enumeration of model deployment stages"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    RETIRED = "retired"
    ARCHIVED = "archived"
    EXPERIMENTAL = "experimental"
    VALIDATION = "validation"

class StageTransition(str, Enum):
    """Valid stage transitions"""
    DEV_TO_TEST = "development_to_testing"
    TEST_TO_STAGING = "testing_to_staging"
    STAGING_TO_PROD = "staging_to_production"
    PROD_TO_RETIRED = "production_to_retired"
    RETIRED_TO_ARCHIVED = "retired_to_archived"
    ANY_TO_EXPERIMENTAL = "any_to_experimental"
    EXPERIMENTAL_TO_DEV = "experimental_to_development"
    ANY_TO_VALIDATION = "any_to_validation"
    VALIDATION_TO_STAGING = "validation_to_staging"

@dataclass
class StageConfig:
    """Configuration for a specific model stage"""
    stage: ModelStage
    requires_approval: bool = False
    max_models: Optional[int] = None
    auto_monitoring: bool = True
    backup_required: bool = False
    performance_thresholds: Dict[str, float] = field(default_factory=dict)
    allowed_transitions: Set[ModelStage] = field(default_factory=set)
    metadata_requirements: List[str] = field(default_factory=list)

@dataclass
class StageTransitionRecord:
    """Record of stage transition"""
    model_id: str
    from_stage: ModelStage
    to_stage: ModelStage
    transition_type: StageTransition
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    reason: Optional[str] = None
    approval_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class StageValidationError(Exception):
    """Exception raised for invalid stage operations"""
    pass

class ModelStageManager:
    """Enterprise-grade model stage management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        
        # Stage configurations
        self.stage_configs = self._initialize_stage_configs(config)
        
        # Transition rules
        self.transition_rules = self._initialize_transition_rules()
        
        # Audit trail
        self.transition_history: List[StageTransitionRecord] = []
        
        # Current stage tracking
        self.model_stages: Dict[str, ModelStage] = {}
        
        # Performance tracking
        self.stage_metrics = {
            stage: {
                'model_count': 0,
                'successful_transitions': 0,
                'failed_transitions': 0,
                'last_transition': None
            }
            for stage in ModelStage
        }
    
    def _initialize_stage_configs(self, config: Dict[str, Any]) -> Dict[ModelStage, StageConfig]:
        """Initialize stage configurations"""
        
        configs = {}
        
        # Development stage
        configs[ModelStage.DEVELOPMENT] = StageConfig(
            stage=ModelStage.DEVELOPMENT,
            requires_approval=False,
            max_models=None,
            auto_monitoring=True,
            backup_required=False,
            performance_thresholds={},
            allowed_transitions={ModelStage.TESTING, ModelStage.EXPERIMENTAL},
            metadata_requirements=['model_type', 'created_by']
        )
        
        # Testing stage
        configs[ModelStage.TESTING] = StageConfig(
            stage=ModelStage.TESTING,
            requires_approval=config.get('testing_requires_approval', False),
            max_models=config.get('max_testing_models', 10),
            auto_monitoring=True,
            backup_required=True,
            performance_thresholds={
                'accuracy': 0.7,
                'precision': 0.65,
                'recall': 0.65
            },
            allowed_transitions={ModelStage.STAGING, ModelStage.DEVELOPMENT, ModelStage.VALIDATION},
            metadata_requirements=['model_type', 'created_by', 'test_results']
        )
        
        # Staging stage
        configs[ModelStage.STAGING] = StageConfig(
            stage=ModelStage.STAGING,
            requires_approval=config.get('staging_requires_approval', True),
            max_models=config.get('max_staging_models', 5),
            auto_monitoring=True,
            backup_required=True,
            performance_thresholds={
                'accuracy': 0.8,
                'precision': 0.75,
                'recall': 0.75,
                'latency_ms': 100
            },
            allowed_transitions={ModelStage.PRODUCTION, ModelStage.TESTING},
            metadata_requirements=['model_type', 'created_by', 'test_results', 'validation_results']
        )
        
        # Production stage
        configs[ModelStage.PRODUCTION] = StageConfig(
            stage=ModelStage.PRODUCTION,
            requires_approval=config.get('production_requires_approval', True),
            max_models=config.get('max_production_models', 3),
            auto_monitoring=True,
            backup_required=True,
            performance_thresholds={
                'accuracy': 0.85,
                'precision': 0.8,
                'recall': 0.8,
                'latency_ms': 50,
                'uptime_percent': 99.9
            },
            allowed_transitions={ModelStage.RETIRED},
            metadata_requirements=['model_type', 'created_by', 'test_results', 'validation_results', 'approval_id']
        )
        
        # Retired stage
        configs[ModelStage.RETIRED] = StageConfig(
            stage=ModelStage.RETIRED,
            requires_approval=False,
            max_models=None,
            auto_monitoring=False,
            backup_required=True,
            performance_thresholds={},
            allowed_transitions={ModelStage.ARCHIVED},
            metadata_requirements=['retirement_reason', 'retired_by']
        )
        
        return configs
    
    def _initialize_transition_rules(self) -> Dict[Tuple[ModelStage, ModelStage], StageTransition]:
        """Initialize valid transition rules"""
        
        return {
            (ModelStage.DEVELOPMENT, ModelStage.TESTING): StageTransition.DEV_TO_TEST,
            (ModelStage.TESTING, ModelStage.STAGING): StageTransition.TEST_TO_STAGING,
            (ModelStage.STAGING, ModelStage.PRODUCTION): StageTransition.STAGING_TO_PROD,
            (ModelStage.PRODUCTION, ModelStage.RETIRED): StageTransition.PROD_TO_RETIRED,
            (ModelStage.RETIRED, ModelStage.ARCHIVED): StageTransition.RETIRED_TO_ARCHIVED,
            (ModelStage.EXPERIMENTAL, ModelStage.DEVELOPMENT): StageTransition.EXPERIMENTAL_TO_DEV,
            (ModelStage.VALIDATION, ModelStage.STAGING): StageTransition.VALIDATION_TO_STAGING,
        }
    
    def get_model_stage(self, model_id: str) -> Optional[ModelStage]:
        """
        Get current stage for a model
        Replace hardcoded 'PRODUCTION' string with enum-based stage lookup
        """
        return self.model_stages.get(model_id)
    
    def set_model_stage(self, model_id: str, stage: ModelStage, 
                       user_id: Optional[str] = None,
                       reason: Optional[str] = None,
                       approval_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Set model stage with validation and audit trail
        Replace string-based stage management with enum-based system
        """
        
        try:
            current_stage = self.model_stages.get(model_id)
            
            # Validate transition if model already has a stage
            if current_stage:
                if not self._validate_stage_transition(current_stage, stage):
                    raise StageValidationError(
                        f"Invalid transition from {current_stage.value} to {stage.value}"
                    )
            
            # Validate stage requirements
            if not self._validate_stage_requirements(model_id, stage, metadata or {}):
                raise StageValidationError(f"Stage requirements not met for {stage.value}")
            
            # Check stage capacity
            if not self._check_stage_capacity(stage):
                raise StageValidationError(f"Stage {stage.value} has reached maximum capacity")
            
            # Check approval requirements
            if self.stage_configs[stage].requires_approval and not approval_id:
                raise StageValidationError(f"Stage {stage.value} requires approval")
            
            # Record transition
            transition_record = StageTransitionRecord(
                model_id=model_id,
                from_stage=current_stage or ModelStage.DEVELOPMENT,
                to_stage=stage,
                transition_type=self._get_transition_type(current_stage, stage),
                user_id=user_id,
                reason=reason,
                approval_id=approval_id,
                metadata=metadata or {}
            )
            
            # Update model stage
            self.model_stages[model_id] = stage
            
            # Record transition
            self.transition_history.append(transition_record)
            
            # Update metrics
            self._update_stage_metrics(stage, success=True)
            
            self.logger.info(f"Model {model_id} transitioned to stage {stage.value}")
            return True
            
        except Exception as e:
            self._update_stage_metrics(stage, success=False)
            self.logger.error(f"Failed to set model {model_id} to stage {stage.value}: {e}")
            raise
    
    def _validate_stage_transition(self, from_stage: ModelStage, to_stage: ModelStage) -> bool:
        """Validate if stage transition is allowed"""
        
        # Check if transition is explicitly allowed
        if to_stage in self.stage_configs[from_stage].allowed_transitions:
            return True
        
        # Check if it's a valid transition rule
        transition_key = (from_stage, to_stage)
        if transition_key in self.transition_rules:
            return True
        
        # Allow transitions to experimental and validation from any stage
        if to_stage in [ModelStage.EXPERIMENTAL, ModelStage.VALIDATION]:
            return True
        
        return False
    
    def _validate_stage_requirements(self, model_id: str, stage: ModelStage, metadata: Dict[str, Any]) -> bool:
        """Validate that stage requirements are met"""
        
        stage_config = self.stage_configs[stage]
        
        # Check required metadata fields
        for required_field in stage_config.metadata_requirements:
            if required_field not in metadata:
                self.logger.error(f"Missing required metadata field '{required_field}' for stage {stage.value}")
                return False
        
        # Check performance thresholds
        for metric, threshold in stage_config.performance_thresholds.items():
            if metric in metadata:
                actual_value = metadata[metric]
                if actual_value < threshold:
                    self.logger.error(f"Performance metric '{metric}' ({actual_value}) below threshold ({threshold}) for stage {stage.value}")
                    return False
        
        return True
    
    def _check_stage_capacity(self, stage: ModelStage) -> bool:
        """Check if stage has capacity for new models"""
        
        stage_config = self.stage_configs[stage]
        
        if stage_config.max_models is None:
            return True
        
        current_count = sum(1 for s in self.model_stages.values() if s == stage)
        
        return current_count < stage_config.max_models
    
    def _get_transition_type(self, from_stage: Optional[ModelStage], to_stage: ModelStage) -> StageTransition:
        """Get the appropriate transition type"""
        
        if not from_stage:
            return StageTransition.ANY_TO_EXPERIMENTAL  # Default for new models
        
        transition_key = (from_stage, to_stage)
        return self.transition_rules.get(transition_key, StageTransition.ANY_TO_EXPERIMENTAL)
    
    def _update_stage_metrics(self, stage: ModelStage, success: bool) -> None:
        """Update stage metrics"""
        
        if success:
            self.stage_metrics[stage]['successful_transitions'] += 1
        else:
            self.stage_metrics[stage]['failed_transitions'] += 1
        
        self.stage_metrics[stage]['last_transition'] = datetime.utcnow()
        
        # Update model count
        self.stage_metrics[stage]['model_count'] = sum(1 for s in self.model_stages.values() if s == stage)
    
    def is_production_model(self, model_id: str) -> bool:
        """
        Check if model is in production stage
        Replace hardcoded 'PRODUCTION' string with enum-based check
        """
        return self.get_model_stage(model_id) == ModelStage.PRODUCTION
    
    def get_production_models(self) -> List[str]:
        """Get all models in production stage"""
        return [
            model_id for model_id, stage in self.model_stages.items()
            if stage == ModelStage.PRODUCTION
        ]
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """Get summary of all stages"""
        
        summary = {}
        
        for stage in ModelStage:
            stage_models = [model_id for model_id, s in self.model_stages.items() if s == stage]
            
            summary[stage.value] = {
                'model_count': len(stage_models),
                'models': stage_models,
                'config': {
                    'requires_approval': self.stage_configs[stage].requires_approval,
                    'max_models': self.stage_configs[stage].max_models,
                    'performance_thresholds': self.stage_configs[stage].performance_thresholds
                },
                'metrics': self.stage_metrics[stage]
            }
        
        return summary
    
    def get_transition_history(self, model_id: Optional[str] = None) -> List[StageTransitionRecord]:
        """Get transition history for a model or all models"""
        
        if model_id:
            return [record for record in self.transition_history if record.model_id == model_id]
        
        return self.transition_history.copy()
    
    def promote_model(self, model_id: str, target_stage: ModelStage, 
                     user_id: Optional[str] = None,
                     approval_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Promote model to next stage with validation"""
        
        current_stage = self.get_model_stage(model_id)
        
        if not current_stage:
            raise StageValidationError(f"Model {model_id} has no current stage")
        
        # Validate promotion path
        if target_stage not in self.stage_configs[current_stage].allowed_transitions:
            raise StageValidationError(
                f"Cannot promote from {current_stage.value} to {target_stage.value}"
            )
        
        return self.set_model_stage(
            model_id=model_id,
            stage=target_stage,
            user_id=user_id,
            reason=f"Promotion from {current_stage.value}",
            approval_id=approval_id,
            metadata=metadata
        )

# Enhanced Model Repository with Stage Management
class EnhancedModelRepository:
    """Enhanced model repository with enum-based stage management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.stage_manager = ModelStageManager(config)
        self.logger = logging.getLogger(__name__)
    
    def is_model_in_production(self, model_id: str) -> bool:
        """
        Check if model is in production
        Replace hardcoded "PRODUCTION" string with enum-based check
        """
        return self.stage_manager.is_production_model(model_id)
    
    def get_production_models(self) -> List[str]:
        """Get all production models using enum-based filtering"""
        return self.stage_manager.get_production_models()
    
    def deploy_to_production(self, model_id: str, user_id: str, 
                           approval_id: str, metadata: Dict[str, Any]) -> bool:
        """Deploy model to production with proper stage management"""
        
        try:
            return self.stage_manager.set_model_stage(
                model_id=model_id,
                stage=ModelStage.PRODUCTION,
                user_id=user_id,
                reason="Production deployment",
                approval_id=approval_id,
                metadata=metadata
            )
        except Exception as e:
            self.logger.error(f"Failed to deploy model {model_id} to production: {e}")
            return False
```

#### c. Key Considerations & Dependencies
- **Type Safety:** Enum-based stage management prevents string literal errors
- **Validation:** Comprehensive stage transition validation and requirements checking
- **Audit Trail:** Complete tracking of stage changes for compliance
- **Configuration:** Flexible stage configuration for different environments

### 4. Acceptance Criteria
- [ ] ModelStage enum replaces all hardcoded string literals
- [ ] Comprehensive stage transition validation system
- [ ] Stage-specific configuration and requirements
- [ ] Audit trail for all stage changes
- [ ] Capacity management for each stage
- [ ] Performance threshold validation
- [ ] Approval workflow integration
- [ ] Stage promotion workflow
- [ ] Production model identification using enum comparison
- [ ] Comprehensive stage monitoring and metrics
- [ ] Hardcoded "PRODUCTION" string completely replaced with enum-based system 