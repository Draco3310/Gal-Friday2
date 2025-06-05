# Provide Real Usage Documentation for Feature Registry Client

## Task ID
**FEATURE-REGISTRY-001**

## Priority
**Medium**

## Epic
**Documentation & Developer Experience**

## Story
As a developer integrating with the feature registry client, I need comprehensive usage documentation and examples so that I can effectively utilize the feature registry system in my applications.

## Problem Statement
Line 37 in `gal_friday/core/feature_registry_client.py` contains placeholder example code that needs to be replaced with real usage documentation or removed entirely. This affects developer understanding and proper integration.

## Acceptance Criteria
- [ ] Review current placeholder example section in the feature registry client
- [ ] Create comprehensive usage documentation with real examples
- [ ] Include common integration patterns and best practices
- [ ] Add error handling examples and troubleshooting guidance
- [ ] Provide code samples for typical feature registry operations
- [ ] Remove placeholder content and replace with production-ready documentation
- [ ] Add API reference documentation for all public methods

## Technical Requirements
- Review line 37 in `gal_friday/core/feature_registry_client.py`
- Create comprehensive docstrings with usage examples
- Document all public methods and their parameters
- Include type hints and return value documentation
- Add examples for common use cases (registration, retrieval, updates)
- Follow Python documentation standards (Google/NumPy style)

## Definition of Done
- [ ] All placeholder example code is removed or converted to proper documentation
- [ ] Comprehensive usage examples are provided for main functionality
- [ ] API documentation covers all public methods and classes
- [ ] Error handling patterns are documented with examples
- [ ] Integration patterns are clearly explained
- [ ] Documentation is reviewed for accuracy and completeness
- [ ] Code review completed and approved

## Dependencies
- Understanding of feature registry architecture and capabilities
- Knowledge of client usage patterns and requirements
- Integration requirements from other system components

## Estimated Effort
**Story Points: 3**

## Risk Assessment
**Low Risk** - Documentation improvement with no runtime changes

## Implementation Notes
```python
# Example of proper documentation structure
class FeatureRegistryClient:
    """Client for interacting with the feature registry system.
    
    This client provides methods for registering, retrieving, and managing
    features in the centralized feature registry.
    
    Examples:
        Basic usage:
        >>> client = FeatureRegistryClient(base_url="http://registry:8080")
        >>> client.register_feature(feature_spec)
        
        Retrieving features:
        >>> features = client.get_features_by_tag("trading")
        >>> feature = client.get_feature_by_name("rsi_14")
    """
```

## Related Files
- `gal_friday/core/feature_registry_client.py` (line 37)
- Feature registry service documentation
- Integration guides and developer documentation
- Example usage in other modules 