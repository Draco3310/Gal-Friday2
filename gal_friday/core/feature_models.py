"""Pydantic models for structured data interchange.

This module defines Pydantic models used for ensuring data contracts,
specifically for feature payloads published by the FeatureEngine and consumed
by downstream services like the PredictionService.

The module supports both required and optional features with comprehensive
validation and schema generation capabilities.
"""
from typing import Any, ClassVar, Dict, Optional, Union  # For RUF012

from pydantic import BaseModel, Field, ConfigDict, field_validator


class FeatureValidationError(Exception):
    """Raised when feature validation fails."""
    
    def __init__(self, feature_name: str, message: str) -> None:
        """Initialize with feature name and error message.
        
        Args:
            feature_name: Name of the feature that failed validation
            message: Detailed error message
        """
        self.feature_name = feature_name
        self.message = message
        super().__init__(f"Feature '{feature_name}': {message}")


class FeatureSpec(BaseModel):
    """Specification for feature validation and schema generation.
    
    This class defines the rules and validation logic for individual features,
    supporting both required and optional features with configurable defaults
    and validation rules.
    
    Attributes:
        name: Unique identifier for the feature
        required: Whether the feature is required (default: True)
        default_value: Default value for optional features when missing
        validation_rules: Custom validation rules (e.g., min/max values)
        description: Human-readable description of the feature
        data_type: Expected data type for the feature value
    """
    
    name: str = Field(..., description="Unique feature identifier")
    required: bool = Field(True, description="Whether feature is required")
    default_value: Optional[Union[float, int, str]] = Field(
        None, description="Default value for optional features"
    )
    validation_rules: Optional[Dict[str, Any]] = Field(
        None, description="Custom validation rules"
    )
    description: Optional[str] = Field(
        None, description="Human-readable feature description"
    )
    data_type: str = Field("float", description="Expected data type")
    
    def validate_feature(self, value: Any) -> Any:
        """Validate a feature value according to its specification.
        
        Args:
            value: The feature value to validate
            
        Returns:
            The validated (and potentially transformed) feature value
            
        Raises:
            FeatureValidationError: If validation fails
        """
        # Handle missing values
        if value is None:
            if self.required:
                raise FeatureValidationError(
                    self.name, 
                    "Required feature is missing or None"
                )
            return self.default_value
        
        # Type[Any] validation
        if self.data_type == "float":
            try:
                value = float(value)
            except (ValueError, TypeError) as e:
                raise FeatureValidationError(
                    self.name,
                    f"Cannot convert to float: {e}"
                ) from e
        elif self.data_type == "int":
            try:
                value = int(value)
            except (ValueError, TypeError) as e:
                raise FeatureValidationError(
                    self.name,
                    f"Cannot convert to int: {e}"
                ) from e
        
        # Apply custom validation rules
        if self.validation_rules:
            self._apply_validation_rules(value)
        
        return value
    
    def _apply_validation_rules(self, value: Union[float, int, str]) -> None:
        """Apply custom validation rules to a feature value.
        
        Args:
            value: The feature value to validate
            
        Raises:
            FeatureValidationError: If validation rules are not met
        """
        rules = self.validation_rules or {}
        
        if "min_value" in rules and value < rules["min_value"]:
            raise FeatureValidationError(
                self.name,
                f"Value {value} below minimum {rules['min_value']}"
            )
        
        if "max_value" in rules and value > rules["max_value"]:
            raise FeatureValidationError(
                self.name,
                f"Value {value} above maximum {rules['max_value']}"
            )
        
        if "allowed_values" in rules and value not in rules["allowed_values"]:
            raise FeatureValidationError(
                self.name,
                f"Value {value} not in allowed values {rules['allowed_values']}"
            )


class FeatureRegistry:
    """Registry for managing feature specifications and validation.
    
    This class provides a centralized way to define, validate, and retrieve
    feature specifications for the feature pipeline.
    """
    
    def __init__(self) -> None:
        """Initialize empty feature registry."""
        self._specs: Dict[str, FeatureSpec] = {}
    
    def register_feature(self, spec: FeatureSpec) -> None:
        """Register a feature specification.
        
        Args:
            spec: The feature specification to register
            
        Raises:
            ValueError: If feature name already exists
        """
        if spec.name in self._specs:
            raise ValueError(f"Feature '{spec.name}' already registered")
        self._specs[spec.name] = spec
    
    def get_spec(self, name: str) -> Optional[FeatureSpec]:
        """Get feature specification by name.
        
        Args:
            name: Name of the feature
            
        Returns:
            FeatureSpec if found, None otherwise
        """
        return self._specs.get(name)
    
    def validate_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a dictionary of features against registered specs.
        
        Args:
            features: Dictionary of feature name -> value pairs
            
        Returns:
            Dictionary of validated and transformed features
            
        Raises:
            FeatureValidationError: If validation fails
        """
        validated = {}
        
        # Validate provided features
        for name, value in features.items():
            spec = self.get_spec(name)
            if spec:
                validated[name] = spec.validate_feature(value)
            else:
                # Unknown feature - pass through with warning
                validated[name] = value
        
        # Check for missing required features
        for name, spec in self._specs.items():
            if spec.required and name not in features:
                raise FeatureValidationError(
                    name,
                    "Required feature missing from input"
                )
            elif not spec.required and name not in features:
                # Add default value for missing optional feature
                validated[name] = spec.default_value
        
        return validated


class PublishedFeaturesV1(BaseModel):
    """Represents the structure of features calculated by the FeatureEngine.

    This model is published as part of a `FeatureEvent` and ensures that the
    features payload is validated and adheres to a defined data contract
    before being published.

    The model supports both required and optional features with comprehensive
    validation. All feature values are expected to be numerical (floats) unless
    explicitly specified otherwise. The field names must correspond to the final
    "flattened" feature keys as generated by `FeatureEngine._calculate_and_publish_features`.

    If `FeatureEngine` calculates a feature that is not defined here, or if a value
    does not conform to the type specification, Pydantic will raise a validation
    error in `FeatureEngine`, preventing the publication of a malformed feature set.
    
    This model should be kept in sync with the features intended for publication,
    typically those defined and activated via the feature registry and application config.
    
    Features can be either required (must be present) or optional (may be missing,
    will use default values). Optional features are indicated by Optional[] type
    annotations and default values.
    """

    # Required features from sample registry
    # From rsi_14_default in sample registry
    rsi_14_default: float = Field(..., 
        description="14-period RSI technical indicator",
        examples=[50.0]
    )

    # From macd_default in sample registry
    # Assuming pandas_ta names MACD columns like: MACD_<fast>_<slow>_<signal>, MACDh..., MACDs...
    # The FeatureEngine's naming logic is base_feature_key + "_" + col_name
    macd_default_macd_12_26_9: float = Field(..., # N815
        description="MACD line (12,26,9 configuration)",
        examples=[0.5]
    )
    macd_default_macdh_12_26_9: float = Field(..., # N815
        description="MACD histogram (12,26,9 configuration)",
        examples=[0.1]
    )
    macd_default_macds_12_26_9: float = Field(..., # N815
        description="MACD signal line (12,26,9 configuration)",
        examples=[0.4]
    )

    # From l2_spread_basic in sample registry
    # _pipeline_compute_l2_spread outputs a DataFrame with "abs_spread" and "pct_spread"
    l2_spread_basic_abs_spread: float = Field(...,
        description="Absolute spread from Level 2 order book",
        examples=[0.01]
    )
    l2_spread_basic_pct_spread: float = Field(...,
        description="Percentage spread from Level 2 order book",
        examples=[0.001]
    )

    # From vwap_trades_60s in sample registry
    vwap_trades_60s: float = Field(...,
        description="Volume Weighted Average Price over 60-second window",
        examples=[30000.0]
    )

    # Optional features with defaults
    # These features may not always be available but provide fallback values
    market_regime_indicator: Optional[str] = Field(
        default="unknown",
        description="Current market regime classification",
        examples=["trending", "ranging", "volatile", "unknown"]
    )
    
    volatility_percentile: Optional[float] = Field(
        default=50.0,
        description="Current volatility as percentile of historical range",
        examples=[75.5]
    )
    
    liquidity_score: Optional[float] = Field(
        default=1.0,
        description="Market liquidity score (0-10 scale)",
        examples=[7.5]
    )

    @field_validator('rsi_14_default')
    @classmethod
    def validate_rsi(cls, v: float) -> float:
        """Validate RSI is within expected range (0-100)."""
        if not 0 <= v <= 100:
            raise ValueError(f"RSI must be between 0 and 100, got {v}")
        return v

    @field_validator('l2_spread_basic_pct_spread')
    @classmethod
    def validate_pct_spread(cls, v: float) -> float:
        """Validate percentage spread is positive."""
        if v < 0:
            raise ValueError(f"Percentage spread must be non-negative, got {v}")
        return v

    @field_validator('vwap_trades_60s')
    @classmethod
    def validate_vwap(cls, v: float) -> float:
        """Validate VWAP is positive."""
        if v <= 0:
            raise ValueError(f"VWAP must be positive, got {v}")
        return v

    @field_validator('volatility_percentile')
    @classmethod
    def validate_volatility_percentile(cls, v: Optional[float]) -> Optional[float]:
        """Validate volatility percentile is within 0-100 range."""
        if v is not None and not 0 <= v <= 100:
            raise ValueError(f"Volatility percentile must be between 0 and 100, got {v}")
        return v

    @field_validator('liquidity_score')
    @classmethod
    def validate_liquidity_score(cls, v: Optional[float]) -> Optional[float]:
        """Validate liquidity score is within 0-10 range."""
        if v is not None and not 0 <= v <= 10:
            raise ValueError(f"Liquidity score must be between 0 and 10, got {v}")
        return v

    @field_validator('market_regime_indicator')
    @classmethod
    def validate_market_regime(cls, v: Optional[str]) -> Optional[str]:
        """Validate market regime indicator is from allowed values."""
        allowed_regimes = {"trending", "ranging", "volatile", "unknown"}
        if v is not None and v not in allowed_regimes:
            raise ValueError(f"Market regime must be one of {allowed_regimes}, got {v}")
        return v

    def get_feature_completeness(self) -> Dict[str, bool]:
        """Get completeness status for all features.
        
        Returns:
            Dictionary mapping feature names to their presence status
        """
        completeness = {}
        for field_name, field_info in self.__class__.model_fields.items():
            value = getattr(self, field_name)
            is_present = value is not None
            completeness[field_name] = is_present
        return completeness

    def get_missing_optional_features(self) -> list[str]:
        """Get list[Any] of optional features that are missing (None/default).
        
        Returns:
            List of optional feature names that are missing
        """
        missing = []
        for field_name, field_info in self.__class__.model_fields.items():
            if not field_info.is_required():
                value = getattr(self, field_name)
                if value is None or value == field_info.default:
                    missing.append(field_name)
        return missing

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                # Required features
                "rsi_14_default": 55.0,
                "macd_default_macd_12_26_9": -0.23, # N815
                "macd_default_macdh_12_26_9": -0.05, # N815
                "macd_default_macds_12_26_9": -0.18, # N815
                "l2_spread_basic_abs_spread": 0.02,
                "l2_spread_basic_pct_spread": 0.0005,
                "vwap_trades_60s": 30010.50,
                # Optional features
                "market_regime_indicator": "trending",
                "volatility_percentile": 75.5,
                "liquidity_score": 7.5,
            },
            "description": "Feature payload with both required and optional features",
            "feature_categories": {
                "required": [
                    "rsi_14_default",
                    "macd_default_macd_12_26_9",
                    "macd_default_macdh_12_26_9", 
                    "macd_default_macds_12_26_9",
                    "l2_spread_basic_abs_spread",
                    "l2_spread_basic_pct_spread",
                    "vwap_trades_60s"
                ],
                "optional": [
                    "market_regime_indicator",
                    "volatility_percentile", 
                    "liquidity_score"
                ]
            }
        }
    )


# Global feature registry instance for application-wide use
feature_registry = FeatureRegistry()

# Register default feature specifications
_default_specs = [
    FeatureSpec(
        name="rsi_14_default",
        required=True,
        default_value=None,
        data_type="float",
        validation_rules={"min_value": 0, "max_value": 100},
        description="14-period RSI technical indicator"
    ),
    FeatureSpec(
        name="market_regime_indicator",
        required=False,
        default_value="unknown",
        data_type="str",
        validation_rules={"allowed_values": ["trending", "ranging", "volatile", "unknown"]},
        description="Current market regime classification"
    ),
    FeatureSpec(
        name="volatility_percentile",
        required=False,
        default_value=50.0,
        data_type="float",
        validation_rules={"min_value": 0, "max_value": 100},
        description="Current volatility as percentile of historical range"
    ),
    FeatureSpec(
        name="liquidity_score", 
        required=False,
        default_value=1.0,
        data_type="float",
        validation_rules={"min_value": 0, "max_value": 10},
        description="Market liquidity score (0-10 scale)"
    ),
]

# Register the default specifications
for spec in _default_specs:
    feature_registry.register_feature(spec)