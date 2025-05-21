# Test Suite Overhaul Plan

## Overview

The Gal-Friday2 project has undergone significant architectural changes, including:

1. Centralization of interfaces in `src/gal_friday/interfaces/`
2. Enhanced prediction service supporting multiple model types
3. Implementation of model-specific predictor classes
4. Introduction of preprocessing pipelines and ensemble capabilities

This plan outlines a comprehensive approach to rebuild the test suite to align with these architectural changes.

## Goals

1. Ensure all components, especially new or refactored ones, have appropriate test coverage
2. Maintain a consistent testing approach across all modules
3. Properly test interface implementations for contract compliance
4. Implement both unit and integration tests
5. Support continuous integration with automated test runs

## Directory Structure

```
tests/
├── conftest.py                      # Shared fixtures and test utilities
├── unit/                            # Unit tests directory
│   ├── interfaces/                  # Interface contract tests
│   ├── predictors/                  # Predictor implementation tests
│   │   ├── test_xgboost_predictor.py
│   │   └── test_sklearn_predictor.py
│   ├── test_prediction_service.py   # Updated for new architecture
│   └── ...                          # Other module unit tests
└── integration/                     # Integration tests
    ├── test_prediction_pipeline.py  # End-to-end prediction pipeline tests
    └── ...                          # Other integration tests
```

## Migration Steps

1. **Review & Categorize Existing Tests**
   - Identify tests that need updating vs. those that can be preserved
   - Move existing tests to the appropriate unit/ or integration/ directory
   - Update imports to match the new architecture

2. **Create Interface Contract Tests**
   - Develop tests that validate interface contracts
   - Test that implementations properly fulfill their interfaces
   - Focus on `PredictorInterface`, `ExecutionHandlerInterface`, `MarketPriceService`, and `HistoricalDataService`

3. **Update Core Service Tests**
   - Refactor `test_prediction_service.py` to test the enhanced capabilities
   - Update other service tests to use the centralized interfaces
   - Ensure proper mocking of dependencies

4. **Create Implementation-Specific Tests**
   - Develop focused tests for `XGBoostPredictor` and `SklearnPredictor`
   - Test preprocessing pipeline functionality
   - Test model loading, reloading, and error handling
   - Test ensemble combination strategies

5. **Develop Integration Tests**
   - Create tests that verify end-to-end functionality
   - Test the prediction pipeline with mock models
   - Test interoperability between refactored components

## Specific Testing Areas

### Interface Contract Tests

Tests should verify that:
- Abstract method signatures are properly implemented
- Method behaviors align with interface contracts
- Inheritance hierarchies are correctly established

### Predictor Tests

For each predictor implementation:
- Test model loading functionality
- Test prediction output formats
- Test preprocessing capabilities
- Test error handling (missing files, invalid models)
- Test feature validation against expected inputs

### Prediction Service Tests

- Test loading and configuration of multiple model types
- Test preprocessing pipeline integration
- Test ensemble strategies for combining predictions
- Test dynamic model reloading
- Test NaN handling and feature quality checks
- Test event publishing

### Integration Tests

- Test prediction service with actual (small) models
- Test integration with feature engine
- Test end-to-end pipelines
- Test configuration-driven behavior

## Testing Utilities & Fixtures

Update `conftest.py` to provide:
- Mock model factories for different model types
- Mock feature data generators
- Common assertion utilities
- Shared fixtures for configuration, events, and services

## Implementation Timeline

1. **Phase 1: Structure & Foundations (Week 1)**
   - Create directory structure
   - Migrate and categorize existing tests
   - Update conftest.py

2. **Phase 2: Interface & Predictor Tests (Week 1-2)**
   - Develop interface contract tests
   - Implement XGBoostPredictor and SklearnPredictor tests

3. **Phase 3: Service Tests (Week 2-3)**
   - Update prediction_service tests
   - Update other service tests for interface usage

4. **Phase 4: Integration Tests (Week 3-4)**
   - Develop end-to-end testing scenarios
   - Create integration test suites

5. **Phase 5: Validation & Coverage Analysis (Week 4)**
   - Run full test suite
   - Analyze coverage reports
   - Address gaps in coverage

## Continuous Integration

- Ensure all tests can run in CI environment
- Configure pytest to generate JUnit XML reports
- Set up coverage reporting
- Add test badges to documentation

## Test Documentation

For each test module:
- Include clear docstrings explaining test purpose
- Document test data and assertions
- Explain test organization and dependencies

This plan provides a framework for comprehensively updating the test suite to match Gal-Friday2's enhanced architecture while ensuring proper test coverage across all components.
