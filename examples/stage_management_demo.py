"""
Demonstration script for enterprise-grade model stage management system.

This script shows how the new enum-based stage management system replaces
hardcoded string literals with proper validation, audit trails, and enterprise controls.
"""

import asyncio
import uuid
from datetime import datetime, UTC
from typing import Any

# Import the stage management components
from gal_friday.model_lifecycle.registry import ModelStage
from gal_friday.dal.repositories.stage_management import (
    ModelStageManager,
    StageValidationError,
    DEFAULT_STAGE_CONFIG,
    StageConfig,
    StageValidationLevel
)


def demonstrate_enum_based_stage_checking():
    """Demonstrate replacement of hardcoded 'PRODUCTION' string with enum-based checking."""
    
    print("=== Enum-Based Stage Checking Demo ===")
    print("Replacing hardcoded 'PRODUCTION' string with enum-based validation\n")
    
    # Initialize stage manager
    config = DEFAULT_STAGE_CONFIG.copy()
    stage_manager = ModelStageManager(config)
    
    # Test cases for stage checking
    test_stages = [
        "production",
        "PRODUCTION", 
        "staging",
        "development",
        "invalid_stage"
    ]
    
    print("Testing stage validation:")
    for stage_str in test_stages:
        try:
            # OLD WAY (hardcoded string comparison):
            old_way_result = stage_str.upper() == "PRODUCTION"
            
            # NEW WAY (enum-based checking):
            new_way_result = stage_manager.is_production_stage(stage_str)
            
            print(f"  Stage '{stage_str}':")
            print(f"    Old hardcoded check: {old_way_result}")
            print(f"    New enum-based check: {new_way_result}")
            print(f"    Results match: {old_way_result == new_way_result}")
            
        except Exception as e:
            print(f"  Stage '{stage_str}': Error with new method - {e}")
            print(f"    Old method would incorrectly return: {stage_str.upper() == 'PRODUCTION'}")
        
        print()


def demonstrate_stage_transition_validation():
    """Demonstrate comprehensive stage transition validation."""
    
    print("=== Stage Transition Validation Demo ===")
    print("Enterprise-grade validation with approval requirements and performance thresholds\n")
    
    # Initialize stage manager with custom config
    config = {
        "production_requires_approval": True,
        "production_min_accuracy": 0.85,
        "production_min_precision": 0.80,
        "staging_min_accuracy": 0.75,
        "max_production_models": 3
    }
    stage_manager = ModelStageManager(config)
    
    model_id = str(uuid.uuid4())
    
    # Test valid transition
    print("1. Testing valid development -> staging transition:")
    metadata = {
        "model_type": "xgboost",
        "created_by": "data_scientist_1",
        "validation_results": "passed",
        "accuracy": 0.78
    }
    
    is_valid, error = stage_manager.validate_stage_transition(
        model_id=model_id,
        from_stage=ModelStage.DEVELOPMENT,
        to_stage=ModelStage.STAGING,
        metadata=metadata
    )
    
    print(f"   Valid: {is_valid}")
    print(f"   Error: {error}")
    print()
    
    # Test invalid transition (missing approval)
    print("2. Testing invalid staging -> production transition (missing approval):")
    metadata_no_approval = {
        "model_type": "xgboost",
        "created_by": "data_scientist_1",
        "validation_results": "passed",
        "accuracy": 0.88,
        "precision": 0.85
    }
    
    is_valid, error = stage_manager.validate_stage_transition(
        model_id=model_id,
        from_stage=ModelStage.STAGING,
        to_stage=ModelStage.PRODUCTION,
        metadata=metadata_no_approval
    )
    
    print(f"   Valid: {is_valid}")
    print(f"   Error: {error}")
    print()
    
    # Test valid production transition with approval
    print("3. Testing valid staging -> production transition (with approval and metrics):")
    metadata_with_approval = {
        "model_type": "xgboost",
        "created_by": "data_scientist_1", 
        "validation_results": "passed",
        "approval_id": "APPROVAL-2024-001",
        "performance_test_results": "passed",
        "accuracy": 0.88,
        "precision": 0.85,
        "recall": 0.82
    }
    
    is_valid, error = stage_manager.validate_stage_transition(
        model_id=model_id,
        from_stage=ModelStage.STAGING,
        to_stage=ModelStage.PRODUCTION,
        metadata=metadata_with_approval
    )
    
    print(f"   Valid: {is_valid}")
    print(f"   Error: {error}")
    print()


def demonstrate_audit_trail():
    """Demonstrate comprehensive audit trail functionality."""
    
    print("=== Audit Trail Demo ===")
    print("Complete audit trail for stage transitions with enterprise-grade logging\n")
    
    stage_manager = ModelStageManager(DEFAULT_STAGE_CONFIG)
    model_id = str(uuid.uuid4())
    
    # Simulate a series of stage transitions
    transitions = [
        {
            "from_stage": None,
            "to_stage": ModelStage.DEVELOPMENT,
            "user_id": "system",
            "reason": "Initial model registration",
            "metadata": {"model_type": "xgboost", "created_by": "data_scientist_1"}
        },
        {
            "from_stage": ModelStage.DEVELOPMENT,
            "to_stage": ModelStage.STAGING,
            "user_id": "data_scientist_1",
            "reason": "Model validation passed",
            "metadata": {"validation_results": "passed", "accuracy": 0.78}
        },
        {
            "from_stage": ModelStage.STAGING,
            "to_stage": ModelStage.PRODUCTION,
            "user_id": "ml_engineer_1",
            "reason": "Production deployment approved",
            "approval_id": "APPROVAL-2024-001",
            "metadata": {
                "approval_id": "APPROVAL-2024-001",
                "performance_test_results": "passed",
                "accuracy": 0.88,
                "precision": 0.85
            }
        }
    ]
    
    print("Recording stage transitions:")
    for i, transition in enumerate(transitions, 1):
        record = stage_manager.record_stage_transition(
            model_id=model_id,
            **transition
        )
        
        print(f"  {i}. {record.from_stage.value if record.from_stage else 'None'} -> {record.to_stage.value}")
        print(f"     User: {record.user_id}")
        print(f"     Reason: {record.reason}")
        print(f"     Timestamp: {record.timestamp}")
        if record.approval_id:
            print(f"     Approval ID: {record.approval_id}")
        print()
    
    # Show audit trail
    print("Complete audit trail:")
    history = stage_manager.get_transition_history(model_id=model_id)
    
    for record in history:
        print(f"  Transition ID: {record.transition_id}")
        print(f"  {record.from_stage.value if record.from_stage else 'None'} -> {record.to_stage.value}")
        print(f"  Type: {record.transition_type.value}")
        print(f"  Success: {record.success}")
        print(f"  Timestamp: {record.timestamp}")
        print()


def demonstrate_stage_configuration():
    """Demonstrate enterprise-grade stage configuration."""
    
    print("=== Stage Configuration Demo ===")
    print("Enterprise-grade configuration with performance thresholds and capacity limits\n")
    
    stage_manager = ModelStageManager(DEFAULT_STAGE_CONFIG)
    
    # Show stage summary
    summary = stage_manager.get_stage_summary()
    
    print("Stage Configuration Summary:")
    for stage_name, stage_info in summary.items():
        print(f"\n  {stage_name.upper()}:")
        print(f"    Requires Approval: {stage_info['config']['requires_approval']}")
        print(f"    Max Models: {stage_info['config']['max_models']}")
        print(f"    Auto Monitoring: {stage_info['config']['auto_monitoring']}")
        print(f"    Validation Level: {stage_info['config']['validation_level']}")
        
        if stage_info['config']['performance_thresholds']:
            print("    Performance Thresholds:")
            for metric, threshold in stage_info['config']['performance_thresholds'].items():
                print(f"      {metric}: {threshold}")
        
        if stage_info['allowed_transitions']:
            print(f"    Allowed Transitions: {', '.join(stage_info['allowed_transitions'])}")


def main():
    """Run all demonstrations."""
    
    print("🚀 Enterprise-Grade Model Stage Management System Demo")
    print("=" * 60)
    print("Demonstrating the replacement of hardcoded 'PRODUCTION' strings")
    print("with comprehensive enum-based stage management.\n")
    
    try:
        demonstrate_enum_based_stage_checking()
        print("\n" + "=" * 60 + "\n")
        
        demonstrate_stage_transition_validation()
        print("\n" + "=" * 60 + "\n")
        
        demonstrate_audit_trail()
        print("\n" + "=" * 60 + "\n")
        
        demonstrate_stage_configuration()
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("\nKey improvements implemented:")
        print("1. ❌ Removed hardcoded 'PRODUCTION' string")
        print("2. ✅ Added enum-based stage validation")
        print("3. ✅ Implemented comprehensive audit trail")
        print("4. ✅ Added enterprise-grade stage configuration")
        print("5. ✅ Built stage transition validation system")
        print("6. ✅ Created performance threshold checking")
        print("7. ✅ Added approval workflow integration")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 