# Optional Feature Handling Implementation

## Overview

This document describes the implementation of optional feature handling in `gal_friday/core/feature_models.py`, which provides a robust system for managing both required and optional features in the feature pipeline.

## Key Components

### 1. FeatureValidationError
Custom exception class for feature validation failures with detailed error messages including feature names.

### 2. FeatureSpec
A Pydantic model that defines specifications for individual features:
- **name**: Unique feature identifier
- **required**: Whether the feature is required (default: True)
- **default_value**: Default value for optional features when missing
- **validation_rules**: Custom validation rules (min/max values, allowed values)
- **description**: Human-readable feature description
- **data_type**: Expected data type (float, int, str)

### 3. FeatureRegistry
Centralized registry for managing feature specifications:
- Register feature specifications
- Validate feature dictionaries against registered specs
- Handle missing required features (raise errors)
- Handle missing optional features (apply defaults)

### 4. Enhanced PublishedFeaturesV1
Updated Pydantic model with:
- **Required features**: All existing features from the sample registry
- **Optional features**: New features with default values
  - `market_regime_indicator`: Market regime classification (default: "unknown")
  - `volatility_percentile`: Volatility percentile (default: 50.0)
  - `liquidity_score`: Market liquidity score (default: 1.0)
- **Field validators**: Comprehensive validation for all features
- **Utility methods**: 
  - `get_feature_completeness()`: Returns feature presence status
  - `get_missing_optional_features()`: Lists missing optional features

## Features

### Required Features (Must be present)
- `rsi_14_default`: 14-period RSI (0-100 range)
- `macd_default_*`: MACD components
- `l2_spread_basic_*`: Level 2 spread metrics
- `vwap_trades_60s`: VWAP over 60-second window

### Optional Features (Have defaults)
- `market_regime_indicator`: Market regime ("trending", "ranging", "volatile", "unknown")
- `volatility_percentile`: Volatility percentile (0-100 range)
- `liquidity_score`: Liquidity score (0-10 range)

## Validation Rules

### Built-in Validations
- **RSI**: Must be between 0-100
- **Percentage Spread**: Must be non-negative
- **VWAP**: Must be positive
- **Volatility Percentile**: Must be between 0-100 (if provided)
- **Liquidity Score**: Must be between 0-10 (if provided)
- **Market Regime**: Must be one of allowed values

### Custom Validation Rules
The `FeatureSpec` class supports custom validation rules:
- `min_value` / `max_value`: Numeric range validation
- `allowed_values`: Categorical value validation

## Usage Examples

### Creating Features with Defaults
```python
# Only required features - optional ones get defaults
features = PublishedFeaturesV1(
    rsi_14_default=65.0,
    macd_default_macd_12_26_9=0.5,
    # ... other required features
)
# features.market_regime_indicator == "unknown" (default)
```

### Using the Registry
```python
from gal_friday.core.feature_models import feature_registry

# Validate features with registry
raw_features = {"rsi_14_default": 65.0, "unknown_feature": 123}
validated = feature_registry.validate_features(raw_features)
# Adds defaults for missing optional features
```

### Feature Completeness Reporting
```python
features = PublishedFeaturesV1(...)
completeness = features.get_feature_completeness()
missing_optional = features.get_missing_optional_features()
```

## Schema Generation

The model generates comprehensive JSON schemas with:
- Feature examples and descriptions
- Validation rules documentation
- Categorization of required vs optional features
- Default value specifications

## Error Handling

The system provides detailed error messages for:
- Missing required features
- Invalid feature values
- Type conversion failures
- Validation rule violations

## Testing

Comprehensive unit tests cover:
- Feature specification creation and validation
- Registry functionality
- Pydantic model validation
- Error scenarios
- Schema generation
- Integration scenarios

All tests pass with 100% coverage of the new functionality.

## Migration Notes

The implementation is fully backward compatible:
- Existing required features remain unchanged
- New optional features have sensible defaults
- Validation is enhanced but non-breaking
- Schema generation is improved with more metadata

## Future Extensibility

The design supports easy extension:
- Add new optional features by updating the model
- Register new feature specifications in the global registry
- Extend validation rules as needed
- Add new data types and validation logic 