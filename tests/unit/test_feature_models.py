"""Unit tests for feature models and validation system.

This module tests the optional feature handling, validation logic,
and schema generation capabilities implemented in gal_friday.core.feature_models.
"""
import pytest
from typing import Dict, Any

from gal_friday.core.feature_models import (
    FeatureSpec,
    FeatureRegistry,
    FeatureValidationError,
    PublishedFeaturesV1,
    feature_registry,
)


class TestFeatureValidationError:
    """Test cases for FeatureValidationError exception."""
    
    def test_error_initialization(self):
        """Test proper initialization of FeatureValidationError."""
        error = FeatureValidationError("test_feature", "test message")
        assert error.feature_name == "test_feature"
        assert error.message == "test message"
        assert str(error) == "Feature 'test_feature': test message"


class TestFeatureSpec:
    """Test cases for FeatureSpec class."""
    
    def test_feature_spec_creation(self):
        """Test creating a basic FeatureSpec."""
        spec = FeatureSpec(name="test_feature")
        assert spec.name == "test_feature"
        assert spec.required is True
        assert spec.default_value is None
        assert spec.validation_rules is None
        assert spec.data_type == "float"
    
    def test_feature_spec_with_all_fields(self):
        """Test creating FeatureSpec with all fields."""
        spec = FeatureSpec(
            name="test_feature",
            required=False,
            default_value=42.0,
            validation_rules={"min_value": 0, "max_value": 100},
            description="Test feature description",
            data_type="float"
        )
        assert spec.name == "test_feature"
        assert spec.required is False
        assert spec.default_value == 42.0
        assert spec.validation_rules == {"min_value": 0, "max_value": 100}
        assert spec.description == "Test feature description"
        assert spec.data_type == "float"
    
    def test_validate_required_feature_missing(self):
        """Test validation fails for missing required feature."""
        spec = FeatureSpec(name="required_feature", required=True)
        with pytest.raises(FeatureValidationError) as exc_info:
            spec.validate_feature(None)
        assert "Required feature is missing or None" in str(exc_info.value)
        assert exc_info.value.feature_name == "required_feature"
    
    def test_validate_optional_feature_missing_returns_default(self):
        """Test optional feature returns default when missing."""
        spec = FeatureSpec(
            name="optional_feature", 
            required=False, 
            default_value=10.0
        )
        result = spec.validate_feature(None)
        assert result == 10.0
    
    def test_validate_float_conversion(self):
        """Test float type conversion."""
        spec = FeatureSpec(name="float_feature", data_type="float")
        assert spec.validate_feature("123.45") == 123.45
        assert spec.validate_feature(42) == 42.0
    
    def test_validate_int_conversion(self):
        """Test int type conversion."""
        spec = FeatureSpec(name="int_feature", data_type="int")
        assert spec.validate_feature("42") == 42
        assert spec.validate_feature(42.7) == 42
    
    def test_validate_invalid_float_conversion(self):
        """Test invalid float conversion raises error."""
        spec = FeatureSpec(name="float_feature", data_type="float")
        with pytest.raises(FeatureValidationError) as exc_info:
            spec.validate_feature("invalid")
        assert "Cannot convert to float" in str(exc_info.value)
    
    def test_validate_min_value_rule(self):
        """Test minimum value validation rule."""
        spec = FeatureSpec(
            name="bounded_feature",
            validation_rules={"min_value": 0}
        )
        assert spec.validate_feature(5.0) == 5.0
        
        with pytest.raises(FeatureValidationError) as exc_info:
            spec.validate_feature(-1.0)
        assert "below minimum" in str(exc_info.value)
    
    def test_validate_max_value_rule(self):
        """Test maximum value validation rule."""
        spec = FeatureSpec(
            name="bounded_feature",
            validation_rules={"max_value": 100}
        )
        assert spec.validate_feature(50.0) == 50.0
        
        with pytest.raises(FeatureValidationError) as exc_info:
            spec.validate_feature(150.0)
        assert "above maximum" in str(exc_info.value)
    
    def test_validate_allowed_values_rule(self):
        """Test allowed values validation rule."""
        spec = FeatureSpec(
            name="categorical_feature",
            data_type="str",
            validation_rules={"allowed_values": ["a", "b", "c"]}
        )
        assert spec.validate_feature("b") == "b"
        
        with pytest.raises(FeatureValidationError) as exc_info:
            spec.validate_feature("d")
        assert "not in allowed values" in str(exc_info.value)


class TestFeatureRegistry:
    """Test cases for FeatureRegistry class."""
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = FeatureRegistry()
        assert len(registry._specs) == 0
    
    def test_register_feature(self):
        """Test registering a feature specification."""
        registry = FeatureRegistry()
        spec = FeatureSpec(name="test_feature")
        registry.register_feature(spec)
        
        retrieved_spec = registry.get_spec("test_feature")
        assert retrieved_spec is not None
        assert retrieved_spec.name == "test_feature"
    
    def test_register_duplicate_feature_raises_error(self):
        """Test registering duplicate feature raises error."""
        registry = FeatureRegistry()
        spec1 = FeatureSpec(name="duplicate_feature")
        spec2 = FeatureSpec(name="duplicate_feature")
        
        registry.register_feature(spec1)
        with pytest.raises(ValueError) as exc_info:
            registry.register_feature(spec2)
        assert "already registered" in str(exc_info.value)
    
    def test_get_nonexistent_spec_returns_none(self):
        """Test getting nonexistent spec returns None."""
        registry = FeatureRegistry()
        assert registry.get_spec("nonexistent") is None
    
    def test_validate_features_with_registered_specs(self):
        """Test validating features with registered specifications."""
        registry = FeatureRegistry()
        registry.register_feature(FeatureSpec(
            name="required_feature",
            required=True,
            validation_rules={"min_value": 0}
        ))
        registry.register_feature(FeatureSpec(
            name="optional_feature",
            required=False,
            default_value=42.0
        ))
        
        # Test with all features present
        features = {"required_feature": 10.0, "optional_feature": 20.0}
        validated = registry.validate_features(features)
        assert validated["required_feature"] == 10.0
        assert validated["optional_feature"] == 20.0
        
        # Test with optional feature missing
        features = {"required_feature": 15.0}
        validated = registry.validate_features(features)
        assert validated["required_feature"] == 15.0
        assert validated["optional_feature"] == 42.0
    
    def test_validate_features_missing_required_raises_error(self):
        """Test validation fails when required feature is missing."""
        registry = FeatureRegistry()
        registry.register_feature(FeatureSpec(name="required_feature", required=True))
        
        features = {"other_feature": 10.0}
        with pytest.raises(FeatureValidationError) as exc_info:
            registry.validate_features(features)
        assert "Required feature missing from input" in str(exc_info.value)
    
    def test_validate_features_with_unknown_feature(self):
        """Test unknown features pass through validation."""
        registry = FeatureRegistry()
        features = {"unknown_feature": 10.0}
        validated = registry.validate_features(features)
        assert validated["unknown_feature"] == 10.0


class TestPublishedFeaturesV1:
    """Test cases for PublishedFeaturesV1 model."""
    
    def test_create_with_required_features_only(self):
        """Test creating model with only required features."""
        features = PublishedFeaturesV1(
            rsi_14_default=65.0,
            macd_default_macd_12_26_9=0.5,
            macd_default_macdh_12_26_9=0.1,
            macd_default_macds_12_26_9=0.4,
            l2_spread_basic_abs_spread=0.01,
            l2_spread_basic_pct_spread=0.001,
            vwap_trades_60s=30000.0
        )
        
        assert features.rsi_14_default == 65.0
        assert features.market_regime_indicator == "unknown"  # default
        assert features.volatility_percentile == 50.0  # default
        assert features.liquidity_score == 1.0  # default
    
    def test_create_with_all_features(self):
        """Test creating model with all features including optional ones."""
        features = PublishedFeaturesV1(
            rsi_14_default=65.0,
            macd_default_macd_12_26_9=0.5,
            macd_default_macdh_12_26_9=0.1,
            macd_default_macds_12_26_9=0.4,
            l2_spread_basic_abs_spread=0.01,
            l2_spread_basic_pct_spread=0.001,
            vwap_trades_60s=30000.0,
            market_regime_indicator="trending",
            volatility_percentile=75.5,
            liquidity_score=8.5
        )
        
        assert features.market_regime_indicator == "trending"
        assert features.volatility_percentile == 75.5
        assert features.liquidity_score == 8.5
    
    def test_rsi_validation(self):
        """Test RSI validation (must be 0-100)."""
        base_features = {
            "macd_default_macd_12_26_9": 0.5,
            "macd_default_macdh_12_26_9": 0.1,
            "macd_default_macds_12_26_9": 0.4,
            "l2_spread_basic_abs_spread": 0.01,
            "l2_spread_basic_pct_spread": 0.001,
            "vwap_trades_60s": 30000.0,
        }
        
        # Valid RSI
        features = PublishedFeaturesV1(rsi_14_default=50.0, **base_features)
        assert features.rsi_14_default == 50.0
        
        # Invalid RSI (too high)
        with pytest.raises(ValueError) as exc_info:
            PublishedFeaturesV1(rsi_14_default=150.0, **base_features)
        assert "RSI must be between 0 and 100" in str(exc_info.value)
        
        # Invalid RSI (too low)
        with pytest.raises(ValueError) as exc_info:
            PublishedFeaturesV1(rsi_14_default=-10.0, **base_features)
        assert "RSI must be between 0 and 100" in str(exc_info.value)
    
    def test_percentage_spread_validation(self):
        """Test percentage spread validation (must be non-negative)."""
        base_features = {
            "rsi_14_default": 50.0,
            "macd_default_macd_12_26_9": 0.5,
            "macd_default_macdh_12_26_9": 0.1,
            "macd_default_macds_12_26_9": 0.4,
            "l2_spread_basic_abs_spread": 0.01,
            "vwap_trades_60s": 30000.0,
        }
        
        # Valid percentage spread
        features = PublishedFeaturesV1(l2_spread_basic_pct_spread=0.001, **base_features)
        assert features.l2_spread_basic_pct_spread == 0.001
        
        # Invalid percentage spread (negative)
        with pytest.raises(ValueError) as exc_info:
            PublishedFeaturesV1(l2_spread_basic_pct_spread=-0.001, **base_features)
        assert "Percentage spread must be non-negative" in str(exc_info.value)
    
    def test_vwap_validation(self):
        """Test VWAP validation (must be positive)."""
        base_features = {
            "rsi_14_default": 50.0,
            "macd_default_macd_12_26_9": 0.5,
            "macd_default_macdh_12_26_9": 0.1,
            "macd_default_macds_12_26_9": 0.4,
            "l2_spread_basic_abs_spread": 0.01,
            "l2_spread_basic_pct_spread": 0.001,
        }
        
        # Valid VWAP
        features = PublishedFeaturesV1(vwap_trades_60s=30000.0, **base_features)
        assert features.vwap_trades_60s == 30000.0
        
        # Invalid VWAP (zero)
        with pytest.raises(ValueError) as exc_info:
            PublishedFeaturesV1(vwap_trades_60s=0.0, **base_features)
        assert "VWAP must be positive" in str(exc_info.value)
        
        # Invalid VWAP (negative)
        with pytest.raises(ValueError) as exc_info:
            PublishedFeaturesV1(vwap_trades_60s=-1000.0, **base_features)
        assert "VWAP must be positive" in str(exc_info.value)
    
    def test_volatility_percentile_validation(self):
        """Test volatility percentile validation (0-100 range)."""
        base_features = {
            "rsi_14_default": 50.0,
            "macd_default_macd_12_26_9": 0.5,
            "macd_default_macdh_12_26_9": 0.1,
            "macd_default_macds_12_26_9": 0.4,
            "l2_spread_basic_abs_spread": 0.01,
            "l2_spread_basic_pct_spread": 0.001,
            "vwap_trades_60s": 30000.0,
        }
        
        # Valid volatility percentile
        features = PublishedFeaturesV1(volatility_percentile=75.5, **base_features)
        assert features.volatility_percentile == 75.5
        
        # Invalid volatility percentile (too high)
        with pytest.raises(ValueError) as exc_info:
            PublishedFeaturesV1(volatility_percentile=150.0, **base_features)
        assert "Volatility percentile must be between 0 and 100" in str(exc_info.value)
    
    def test_liquidity_score_validation(self):
        """Test liquidity score validation (0-10 range)."""
        base_features = {
            "rsi_14_default": 50.0,
            "macd_default_macd_12_26_9": 0.5,
            "macd_default_macdh_12_26_9": 0.1,
            "macd_default_macds_12_26_9": 0.4,
            "l2_spread_basic_abs_spread": 0.01,
            "l2_spread_basic_pct_spread": 0.001,
            "vwap_trades_60s": 30000.0,
        }
        
        # Valid liquidity score
        features = PublishedFeaturesV1(liquidity_score=8.5, **base_features)
        assert features.liquidity_score == 8.5
        
        # Invalid liquidity score (too high)
        with pytest.raises(ValueError) as exc_info:
            PublishedFeaturesV1(liquidity_score=15.0, **base_features)
        assert "Liquidity score must be between 0 and 10" in str(exc_info.value)
    
    def test_market_regime_validation(self):
        """Test market regime indicator validation."""
        base_features = {
            "rsi_14_default": 50.0,
            "macd_default_macd_12_26_9": 0.5,
            "macd_default_macdh_12_26_9": 0.1,
            "macd_default_macds_12_26_9": 0.4,
            "l2_spread_basic_abs_spread": 0.01,
            "l2_spread_basic_pct_spread": 0.001,
            "vwap_trades_60s": 30000.0,
        }
        
        # Valid market regimes
        for regime in ["trending", "ranging", "volatile", "unknown"]:
            features = PublishedFeaturesV1(market_regime_indicator=regime, **base_features)
            assert features.market_regime_indicator == regime
        
        # Invalid market regime
        with pytest.raises(ValueError) as exc_info:
            PublishedFeaturesV1(market_regime_indicator="invalid", **base_features)
        assert "Market regime must be one of" in str(exc_info.value)
    
    def test_get_feature_completeness(self):
        """Test feature completeness reporting."""
        features = PublishedFeaturesV1(
            rsi_14_default=50.0,
            macd_default_macd_12_26_9=0.5,
            macd_default_macdh_12_26_9=0.1,
            macd_default_macds_12_26_9=0.4,
            l2_spread_basic_abs_spread=0.01,
            l2_spread_basic_pct_spread=0.001,
            vwap_trades_60s=30000.0,
            volatility_percentile=None  # Explicitly set to None
        )
        
        completeness = features.get_feature_completeness()
        assert completeness["rsi_14_default"] is True
        assert completeness["volatility_percentile"] is False
    
    def test_get_missing_optional_features(self):
        """Test getting list of missing optional features."""
        features = PublishedFeaturesV1(
            rsi_14_default=50.0,
            macd_default_macd_12_26_9=0.5,
            macd_default_macdh_12_26_9=0.1,
            macd_default_macds_12_26_9=0.4,
            l2_spread_basic_abs_spread=0.01,
            l2_spread_basic_pct_spread=0.001,
            vwap_trades_60s=30000.0,
        )
        
        missing = features.get_missing_optional_features()
        # Should include optional features that are using default values
        expected_missing = ["market_regime_indicator", "volatility_percentile", "liquidity_score"]
        assert all(feature in missing for feature in expected_missing)
    
    def test_schema_generation(self):
        """Test that schema generation includes all features and metadata."""
        schema = PublishedFeaturesV1.model_json_schema()
        
        # Check that all required features are present
        properties = schema["properties"]
        required_features = [
            "rsi_14_default",
            "macd_default_macd_12_26_9",
            "macd_default_macdh_12_26_9",
            "macd_default_macds_12_26_9",
            "l2_spread_basic_abs_spread",
            "l2_spread_basic_pct_spread",
            "vwap_trades_60s"
        ]
        
        for feature in required_features:
            assert feature in properties
        
        # Check that optional features are present with defaults
        optional_features = [
            "market_regime_indicator",
            "volatility_percentile",
            "liquidity_score"
        ]
        
        for feature in optional_features:
            assert feature in properties
            assert "default" in properties[feature]


class TestGlobalFeatureRegistry:
    """Test cases for the global feature registry."""
    
    def test_global_registry_has_default_specs(self):
        """Test that global registry has default feature specifications."""
        # Test that some default specs are registered
        rsi_spec = feature_registry.get_spec("rsi_14_default")
        assert rsi_spec is not None
        assert rsi_spec.required is True
        assert rsi_spec.validation_rules == {"min_value": 0, "max_value": 100}
        
        market_regime_spec = feature_registry.get_spec("market_regime_indicator")
        assert market_regime_spec is not None
        assert market_regime_spec.required is False
        assert market_regime_spec.default_value == "unknown"
    
    def test_global_registry_validation(self):
        """Test validation using the global registry."""
        features = {
            "rsi_14_default": 65.0,
            "market_regime_indicator": "trending"
        }
        
        validated = feature_registry.validate_features(features)
        assert validated["rsi_14_default"] == 65.0
        assert validated["market_regime_indicator"] == "trending"
        # Should have default for missing optional features
        assert validated["volatility_percentile"] == 50.0
        assert validated["liquidity_score"] == 1.0


# Integration test combining multiple components
class TestFeatureModelIntegration:
    """Integration tests for the complete feature model system."""
    
    def test_end_to_end_feature_validation(self):
        """Test complete end-to-end feature validation flow."""
        # Create a custom registry for this test
        test_registry = FeatureRegistry()
        
        # Register some test specs
        test_registry.register_feature(FeatureSpec(
            name="price",
            required=True,
            validation_rules={"min_value": 0},
            description="Asset price"
        ))
        
        test_registry.register_feature(FeatureSpec(
            name="volume",
            required=False,
            default_value=0.0,
            validation_rules={"min_value": 0},
            description="Trading volume"
        ))
        
        # Test successful validation
        raw_features = {"price": "100.50"}  # String that should convert to float
        validated = test_registry.validate_features(raw_features)
        
        assert validated["price"] == 100.50
        assert validated["volume"] == 0.0  # Default value
        
        # Test validation failure
        with pytest.raises(FeatureValidationError):
            test_registry.validate_features({"price": -50.0})  # Below minimum
    
    def test_published_features_with_registry_validation(self):
        """Test that PublishedFeaturesV1 works with registry validation."""
        # Create valid feature data
        feature_data = {
            "rsi_14_default": 65.0,
            "macd_default_macd_12_26_9": 0.5,
            "macd_default_macdh_12_26_9": 0.1,
            "macd_default_macds_12_26_9": 0.4,
            "l2_spread_basic_abs_spread": 0.01,
            "l2_spread_basic_pct_spread": 0.001,
            "vwap_trades_60s": 30000.0,
            "market_regime_indicator": "trending",
            "volatility_percentile": 75.0,
            "liquidity_score": 8.0
        }
        
        # Should work with both Pydantic validation and registry validation
        features = PublishedFeaturesV1(**feature_data)
        validated = feature_registry.validate_features(feature_data)
        
        # Both should produce consistent results
        assert features.rsi_14_default == validated["rsi_14_default"]
        assert features.market_regime_indicator == validated["market_regime_indicator"] 