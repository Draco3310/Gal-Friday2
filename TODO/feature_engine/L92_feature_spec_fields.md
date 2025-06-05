# Task: Add additional fields from FeatureSpec (e.g., output names) and handle multiple outputs properly.

### 1. Context
- **File:** `gal_friday/feature_engine.py`
- **Line:** `92-93`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO comment indicating that additional fields from FeatureSpec need to be added and multiple outputs need proper handling.

### 2. Problem Statement
The current implementation of the FeatureEngine does not fully utilize the FeatureSpec configuration, missing critical fields like output names and proper multiple output handling. This limitation restricts the system's ability to generate complex features with multiple outputs, properly name feature columns, and maintain feature metadata consistency.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Extend FeatureSpec Model:** Add comprehensive fields for output names, types, and metadata
2. **Implement Multiple Output Handling:** Support features that generate multiple output columns
3. **Add Output Naming Strategy:** Flexible naming conventions for feature outputs
4. **Create Output Validation:** Ensure generated outputs match specifications

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from enum import Enum
import pandas as pd

class OutputType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"

class OutputSpec(BaseModel):
    name: str
    output_type: OutputType
    description: Optional[str] = None
    validation_range: Optional[tuple] = None

class EnhancedFeatureSpec(BaseModel):
    name: str
    input_columns: List[str]
    output_specs: List[OutputSpec] = Field(description="Detailed output specifications")
    output_naming_pattern: Optional[str] = Field(default=None)
    computation_function: str
    parameters: Dict[str, Any] = {}
    
    @property
    def output_names(self) -> List[str]:
        if self.output_naming_pattern:
            return [
                self.output_naming_pattern.format(
                    feature_name=self.name,
                    output_name=spec.name
                )
                for spec in self.output_specs
            ]
        return [spec.name for spec in self.output_specs]

def process_feature_outputs(raw_outputs: Union[pd.Series, pd.DataFrame, dict], 
                          feature_spec: EnhancedFeatureSpec) -> pd.DataFrame:
    """Process raw feature outputs according to specification"""
    
    # Convert to DataFrame format
    if isinstance(raw_outputs, pd.Series):
        outputs = pd.DataFrame({feature_spec.output_specs[0].name: raw_outputs})
    elif isinstance(raw_outputs, dict):
        outputs = pd.DataFrame(raw_outputs)
    else:
        outputs = pd.DataFrame(raw_outputs)
    
    # Apply type conversions
    for spec in feature_spec.output_specs:
        if spec.name in outputs.columns:
            if spec.output_type == OutputType.NUMERIC:
                outputs[spec.name] = pd.to_numeric(outputs[spec.name], errors='coerce')
            elif spec.output_type == OutputType.CATEGORICAL:
                outputs[spec.name] = outputs[spec.name].astype('category')
            elif spec.output_type == OutputType.BOOLEAN:
                outputs[spec.name] = outputs[spec.name].astype(bool)
    
    # Apply naming pattern
    if feature_spec.output_naming_pattern:
        rename_map = {}
        for spec in feature_spec.output_specs:
            if spec.name in outputs.columns:
                new_name = feature_spec.output_naming_pattern.format(
                    feature_name=feature_spec.name,
                    output_name=spec.name
                )
                rename_map[spec.name] = new_name
        outputs = outputs.rename(columns=rename_map)
    
    return outputs
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Validation of output structure and types; graceful handling of missing outputs
- **Configuration:** Flexible output naming patterns; configurable validation rules
- **Testing:** Unit tests for output type validation; integration tests with feature computations
- **Dependencies:** Pydantic for data validation; pandas for data manipulation

### 4. Acceptance Criteria
- [ ] EnhancedFeatureSpec model includes output specifications and metadata fields
- [ ] Multiple output handling supports DataFrame, Series, and dictionary formats
- [ ] Output naming patterns provide flexible column naming strategies
- [ ] Type validation works correctly for numeric, categorical, and boolean outputs
- [ ] TODO comments at lines 92-93 are replaced with production implementation 