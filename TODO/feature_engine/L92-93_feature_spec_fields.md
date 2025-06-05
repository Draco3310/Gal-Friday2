# Task: Add additional fields from FeatureSpec (e.g., output names) and handle multiple outputs properly.

### 1. Context
- **File:** `gal_friday/feature_engine.py`
- **Line:** `92-93`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO comment indicating that additional fields from FeatureSpec need to be added and multiple outputs need proper handling.

### 2. Problem Statement
The current implementation of the FeatureEngine does not fully utilize the FeatureSpec configuration, missing critical fields like output names and proper multiple output handling. This limitation restricts the system's ability to generate complex features with multiple outputs, properly name feature columns, and maintain feature metadata consistency. Without proper output handling, feature engineering becomes inflexible and difficult to maintain or debug.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Extend FeatureSpec Model:** Add comprehensive fields for output names, types, and metadata
2. **Implement Multiple Output Handling:** Support features that generate multiple output columns
3. **Add Output Naming Strategy:** Flexible naming conventions for feature outputs
4. **Create Output Validation:** Ensure generated outputs match specifications
5. **Build Metadata Management:** Track feature lineage and transformation details
6. **Add Configuration Flexibility:** Support dynamic feature specifications

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Union, Any, Callable
from pydantic import BaseModel, Field, validator
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import numpy as np

class OutputType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    STRING = "string"

class AggregationMethod(str, Enum):
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    STD = "std"
    COUNT = "count"
    MEDIAN = "median"
    QUANTILE = "quantile"

@dataclass
class OutputSpec:
    """Specification for a single feature output"""
    name: str
    output_type: OutputType
    description: Optional[str] = None
    validation_range: Optional[tuple] = None
    nullable: bool = True
    default_value: Optional[Any] = None

class EnhancedFeatureSpec(BaseModel):
    """Enhanced feature specification with comprehensive output handling"""
    
    # Basic feature information
    name: str = Field(description="Feature name/identifier")
    description: Optional[str] = Field(description="Human-readable feature description")
    category: Optional[str] = Field(description="Feature category (technical, fundamental, sentiment)")
    
    # Input specifications
    input_columns: List[str] = Field(description="Required input columns")
    optional_columns: List[str] = Field(default_factory=list, description="Optional input columns")
    
    # Output specifications (NEW/ENHANCED)
    output_specs: List[OutputSpec] = Field(description="Detailed output specifications")
    output_naming_pattern: Optional[str] = Field(
        default=None, 
        description="Pattern for naming outputs (e.g., '{feature_name}_{output_name}')"
    )
    
    # Computation specifications
    computation_function: str = Field(description="Function name or reference")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Feature parameters")
    
    # Dependencies and requirements
    dependencies: List[str] = Field(default_factory=list, description="Other features this depends on")
    required_lookback_periods: int = Field(default=1, description="Minimum data points required")
    
    # Metadata and versioning (NEW)
    version: str = Field(default="1.0", description="Feature version")
    author: Optional[str] = Field(description="Feature author/creator")
    created_at: Optional[str] = Field(description="Creation timestamp")
    tags: List[str] = Field(default_factory=list, description="Feature tags for organization")
    
    # Performance and caching (NEW)
    cache_enabled: bool = Field(default=True, description="Whether to cache feature results")
    cache_ttl_minutes: Optional[int] = Field(description="Cache time-to-live in minutes")
    computation_priority: int = Field(default=5, description="Computation priority (1-10)")
    
    @validator('output_specs')
    def validate_output_specs(cls, v):
        if not v:
            raise ValueError("At least one output specification is required")
        
        # Check for duplicate output names
        names = [spec.name for spec in v]
        if len(names) != len(set(names)):
            raise ValueError("Output names must be unique")
        
        return v
    
    @property
    def output_names(self) -> List[str]:
        """Get list of all output names"""
        if self.output_naming_pattern:
            return [
                self.output_naming_pattern.format(
                    feature_name=self.name,
                    output_name=spec.name
                )
                for spec in self.output_specs
            ]
        return [spec.name for spec in self.output_specs]
    
    @property
    def expected_output_count(self) -> int:
        """Get expected number of outputs"""
        return len(self.output_specs)

class FeatureOutputHandler:
    """Enhanced handler for multiple feature outputs"""
    
    def __init__(self, feature_spec: EnhancedFeatureSpec):
        self.spec = feature_spec
        self.logger = logging.getLogger(__name__)
        
    def process_feature_outputs(self, raw_outputs: Union[pd.Series, pd.DataFrame, np.ndarray, 
                                                       Dict[str, Any], List[Any]]) -> pd.DataFrame:
        """
        Process raw feature computation outputs according to specification
        Replace/enhance TODO section at lines 92-93
        """
        
        try:
            # Convert raw outputs to standardized format
            standardized_outputs = self._standardize_raw_outputs(raw_outputs)
            
            # Validate output count and structure
            self._validate_output_structure(standardized_outputs)
            
            # Apply output specifications
            processed_outputs = self._apply_output_specifications(standardized_outputs)
            
            # Generate proper column names
            final_outputs = self._apply_output_naming(processed_outputs)
            
            # Validate final outputs
            self._validate_final_outputs(final_outputs)
            
            # Add metadata
            final_outputs = self._add_output_metadata(final_outputs)
            
            self.logger.debug(
                f"Successfully processed {len(final_outputs.columns)} outputs "
                f"for feature {self.spec.name}"
            )
            
            return final_outputs
            
        except Exception as e:
            self.logger.error(f"Error processing outputs for feature {self.spec.name}: {e}")
            raise FeatureProcessingError(f"Failed to process feature outputs: {e}")
    
    def _standardize_raw_outputs(self, raw_outputs: Union[pd.Series, pd.DataFrame, np.ndarray, 
                                                        Dict[str, Any], List[Any]]) -> pd.DataFrame:
        """Convert various output formats to standardized DataFrame"""
        
        if isinstance(raw_outputs, pd.DataFrame):
            return raw_outputs
        
        elif isinstance(raw_outputs, pd.Series):
            # Single series output
            return pd.DataFrame({self.spec.output_specs[0].name: raw_outputs})
        
        elif isinstance(raw_outputs, np.ndarray):
            # NumPy array - could be 1D or 2D
            if raw_outputs.ndim == 1:
                return pd.DataFrame({self.spec.output_specs[0].name: raw_outputs})
            else:
                # Multi-dimensional array
                columns = [spec.name for spec in self.spec.output_specs[:raw_outputs.shape[1]]]
                return pd.DataFrame(raw_outputs, columns=columns)
        
        elif isinstance(raw_outputs, dict):
            # Dictionary of outputs
            return pd.DataFrame(raw_outputs)
        
        elif isinstance(raw_outputs, list):
            # List of values
            if len(self.spec.output_specs) == 1:
                return pd.DataFrame({self.spec.output_specs[0].name: raw_outputs})
            else:
                # Multiple outputs - assume each list element corresponds to an output
                output_dict = {}
                for i, spec in enumerate(self.spec.output_specs):
                    if i < len(raw_outputs):
                        output_dict[spec.name] = raw_outputs[i]
                return pd.DataFrame(output_dict)
        
        else:
            # Single scalar value
            return pd.DataFrame({self.spec.output_specs[0].name: [raw_outputs]})
    
    def _validate_output_structure(self, outputs: pd.DataFrame) -> None:
        """Validate that outputs match expected structure"""
        
        expected_count = self.spec.expected_output_count
        actual_count = len(outputs.columns)
        
        if actual_count != expected_count:
            # Try to handle common mismatches
            if actual_count < expected_count:
                self.logger.warning(
                    f"Feature {self.spec.name} produced {actual_count} outputs, "
                    f"expected {expected_count}. Adding default values."
                )
                # Add missing columns with default values
                for i in range(actual_count, expected_count):
                    spec = self.spec.output_specs[i]
                    default_val = spec.default_value if spec.default_value is not None else np.nan
                    outputs[spec.name] = default_val
            
            elif actual_count > expected_count:
                self.logger.warning(
                    f"Feature {self.spec.name} produced {actual_count} outputs, "
                    f"expected {expected_count}. Truncating to expected count."
                )
                # Keep only expected number of columns
                expected_columns = [spec.name for spec in self.spec.output_specs]
                outputs = outputs[expected_columns[:expected_count]]
    
    def _apply_output_specifications(self, outputs: pd.DataFrame) -> pd.DataFrame:
        """Apply type conversions and validations according to output specs"""
        
        processed = outputs.copy()
        
        for spec in self.spec.output_specs:
            if spec.name not in processed.columns:
                continue
            
            column = processed[spec.name]
            
            # Apply type conversion
            if spec.output_type == OutputType.NUMERIC:
                processed[spec.name] = pd.to_numeric(column, errors='coerce')
            
            elif spec.output_type == OutputType.CATEGORICAL:
                processed[spec.name] = column.astype('category')
            
            elif spec.output_type == OutputType.BOOLEAN:
                processed[spec.name] = column.astype(bool)
            
            elif spec.output_type == OutputType.TIMESTAMP:
                processed[spec.name] = pd.to_datetime(column, errors='coerce')
            
            elif spec.output_type == OutputType.STRING:
                processed[spec.name] = column.astype(str)
            
            # Apply validation range
            if spec.validation_range and spec.output_type == OutputType.NUMERIC:
                min_val, max_val = spec.validation_range
                out_of_range = (column < min_val) | (column > max_val)
                if out_of_range.any():
                    self.logger.warning(
                        f"Feature {self.spec.name} output {spec.name} has "
                        f"{out_of_range.sum()} values outside range [{min_val}, {max_val}]"
                    )
                    # Clip values to range
                    processed[spec.name] = column.clip(min_val, max_val)
            
            # Handle nullability
            if not spec.nullable and column.isnull().any():
                if spec.default_value is not None:
                    processed[spec.name] = column.fillna(spec.default_value)
                else:
                    raise FeatureValidationError(
                        f"Feature {self.spec.name} output {spec.name} contains null values "
                        f"but nullability is disabled"
                    )
        
        return processed
    
    def _apply_output_naming(self, outputs: pd.DataFrame) -> pd.DataFrame:
        """Apply naming pattern to output columns"""
        
        if not self.spec.output_naming_pattern:
            return outputs
        
        renamed_outputs = outputs.copy()
        old_to_new_names = {}
        
        for i, spec in enumerate(self.spec.output_specs):
            if spec.name in outputs.columns:
                new_name = self.spec.output_naming_pattern.format(
                    feature_name=self.spec.name,
                    output_name=spec.name,
                    index=i
                )
                old_to_new_names[spec.name] = new_name
        
        return renamed_outputs.rename(columns=old_to_new_names)
    
    def _add_output_metadata(self, outputs: pd.DataFrame) -> pd.DataFrame:
        """Add metadata attributes to output DataFrame"""
        
        # Add feature metadata as DataFrame attributes
        outputs.attrs['feature_name'] = self.spec.name
        outputs.attrs['feature_version'] = self.spec.version
        outputs.attrs['output_count'] = len(outputs.columns)
        outputs.attrs['computation_timestamp'] = pd.Timestamp.now()
        
        if self.spec.tags:
            outputs.attrs['tags'] = self.spec.tags
        
        return outputs

class EnhancedFeatureEngine:
    """Enhanced feature engine with comprehensive output handling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_specs: Dict[str, EnhancedFeatureSpec] = {}
        self.output_handlers: Dict[str, FeatureOutputHandler] = {}
        
    def register_feature(self, feature_spec: EnhancedFeatureSpec) -> None:
        """Register a feature with enhanced specification"""
        
        self.feature_specs[feature_spec.name] = feature_spec
        self.output_handlers[feature_spec.name] = FeatureOutputHandler(feature_spec)
        
        self.logger.info(
            f"Registered feature {feature_spec.name} with {feature_spec.expected_output_count} outputs"
        )
    
    def compute_feature(self, feature_name: str, input_data: pd.DataFrame) -> pd.DataFrame:
        """Compute feature with enhanced output processing"""
        
        if feature_name not in self.feature_specs:
            raise ValueError(f"Feature {feature_name} not registered")
        
        spec = self.feature_specs[feature_name]
        handler = self.output_handlers[feature_name]
        
        # Get computation function
        compute_func = self._get_computation_function(spec.computation_function)
        
        # Prepare input data
        input_subset = self._prepare_input_data(input_data, spec)
        
        # Compute raw outputs
        raw_outputs = compute_func(input_subset, **spec.parameters)
        
        # Process outputs according to specification
        processed_outputs = handler.process_feature_outputs(raw_outputs)
        
        return processed_outputs
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Comprehensive validation of output structure and types; graceful handling of missing or malformed outputs; fallback strategies for computation failures
- **Configuration:** Flexible output naming patterns; configurable validation rules; runtime specification updates; per-feature caching strategies
- **Testing:** Unit tests for each output type and validation rule; integration tests with real feature computations; performance tests for large feature sets; edge case testing
- **Dependencies:** Pydantic for data validation; pandas for data manipulation; NumPy for numerical operations; logging framework for debugging

### 4. Acceptance Criteria
- [ ] EnhancedFeatureSpec model includes comprehensive output specifications and metadata fields
- [ ] Multiple output handling supports various data formats (DataFrame, Series, arrays, dictionaries)
- [ ] Output naming patterns provide flexible column naming strategies
- [ ] Type validation and conversion work correctly for all supported output types
- [ ] Validation ranges and nullability constraints are properly enforced
- [ ] Feature metadata is preserved and accessible throughout the processing pipeline
- [ ] Performance impact is minimal compared to original implementation
- [ ] Comprehensive test suite covers all output scenarios and edge cases
- [ ] Documentation explains enhanced feature specification format
- [ ] TODO comments at lines 92-93 are replaced with production implementation 