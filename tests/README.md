# Gal-Friday Test Suite

This directory contains the test suite for the Gal-Friday2 cryptocurrency trading system, organized to ensure code quality, correctness and maintainability.

## Test Structure

The test suite is organized into the following structure:

```
tests/
├── unit/                # Unit tests for individual components
│   ├── core/            # Tests for core system components
│   ├── interfaces/      # Interface contract tests
│   ├── market_price/    # Tests for market price services
│   ├── predictors/      # Tests for prediction models
│   ├── execution/       # Tests for execution handler
│   ├── risk/            # Tests for risk management
│   ├── portfolio/       # Tests for portfolio management
│   ├── strategy/        # Tests for strategy components
│   ├── backtesting/     # Tests for backtesting engine
│   └── data/            # Tests for data ingestion and processing
├── integration/         # Integration tests for component interactions
│   └── flows/           # Tests for end-to-end system flows
└── conftest.py          # Shared pytest fixtures
```

## Test Categories

### Unit Tests

Unit tests verify that individual components work correctly in isolation. These tests are fast and focused, helping to pinpoint issues in specific components.

### Interface Contract Tests

Interface contract tests ensure that all implementations of an interface correctly adhere to the contract defined by that interface. These tests help maintain consistency across different implementations of the same interface.

### Integration Tests

Integration tests verify that different components work correctly together. These tests focus on the interactions between components and ensure that the system works as a whole.

## Import Guidelines

When writing tests, certain imports need to come from specific locations:

### Event Classes

**Always import event classes from `gal_friday.event_bus`, not from `gal_friday.core.events`:**

```python
# Correct imports
from gal_friday.event_bus import (
    MarketDataEvent,
    FillEvent,
    OrderEvent,
    SignalEvent
)

# Incorrect imports - these will cause mypy errors
from gal_friday.core.events import MarketDataEvent  # Wrong!
```

### Backpressure Classes

Backpressure strategy classes should be imported from `gal_friday.event_bus`:

```python
# Correct imports
from gal_friday.event_bus import (
    BackpressureStrategy,
    SimpleThresholdBackpressure
)

# Incorrect imports
from gal_friday.core.pubsub import BackpressureStrategy  # Wrong!
```

## Running Tests

### Prerequisites

Ensure you have all dependencies installed:

```bash
pip install -r requirements.txt
pip install pytest pytest-cov pytest-asyncio pytest-xdist
```

### Running All Tests

To run the entire test suite:

```bash
pytest
```

### Running Specific Test Categories

To run only unit tests:

```bash
pytest -m unit

# Or by directory
pytest tests/unit/
```

To run integration tests:

```bash
pytest -m integration
```

To run tests with coverage reporting:

```bash
pytest --cov=gal_friday --cov-report=term --cov-report=xml:coverage.xml
```

### Test Performance Monitoring

For memory-sensitive components, use memory profiling:

```bash
python -m memory_profiler gal_friday/execution_handler.py
```
```

To run interface contract tests:

```bash
pytest tests/unit/interfaces/
```

To run integration tests:

```bash
pytest tests/integration/
```

### Running Tests with Coverage

To run tests with coverage reporting:

```bash
pytest --cov=src tests/
```

For a detailed HTML coverage report:

```bash
pytest --cov=src --cov-report=html tests/
```

The HTML report will be generated in the `htmlcov/` directory.

### Running Tests in Parallel

To speed up test execution, you can run tests in parallel:

```bash
pytest -xvs -n auto tests/
```

## Continuous Integration

The test suite is automatically run in CI for every pull request and push to main branches. The CI pipeline includes:

1. Running all tests across multiple Python versions
2. Coverage reporting
3. Code quality checks (linting, type checking, formatting)
4. Security scanning

## Writing New Tests

When adding new features or fixing bugs, please ensure:

1. Unit tests are added for new components
2. Interface contract tests are added for new interfaces
3. Integration tests are updated if component interactions change
4. Existing tests pass with the new changes

### Test Naming Conventions

- Test files should be named `test_*.py`
- Test functions should be named `test_*`
- Test classes should be named `Test*`

## Mocking and Fixtures

Shared fixtures are defined in `conftest.py`. Use these fixtures whenever possible to maintain consistency across tests.

For external dependencies, use appropriate mocking to ensure tests can run without external services.

To run all tests:

```bash
python -m pytest
```

To run specific tests with verbose output:

```bash
python -m pytest tests/unit/market_price/test_market_price_service.py -v
```

To run type checking on the test suite:

```bash
mypy tests
```
