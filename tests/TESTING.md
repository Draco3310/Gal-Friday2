# Gal-Friday Testing and Code Quality Guide

This document outlines the testing and code quality tools used in the Gal-Friday project and how to use them effectively.

## Code Quality Tools

### Pre-commit Hooks

We use pre-commit hooks to enforce code quality standards before commits are made. To install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

To manually run all pre-commit hooks:

```bash
pre-commit run --all-files
```

### Current Tools

1. **Ruff** - Fast, modern Python linter and formatter that replaces multiple tools (flake8, black, isort, pydocstyle, etc.)
   - Handles code formatting (formerly Black)
   - Manages import sorting (formerly isort)
   - Enforces style guidelines (formerly flake8)
   - Checks docstrings (formerly pydocstyle)
   - Detects commented-out code (ERA rules)
   - Performs security scanning (S rules)
   - Manages complexity (McCabe complexity)
   - And many more code quality checks

2. **mypy** - Static type checker with strict type enforcement

3. **Bandit** - Dedicated security vulnerability scanner

## Testing

We use pytest for unit and integration testing. Tests are located in the `tests/` directory.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/gal_friday tests/

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -xvs

# Include slow tests (disabled by default)
pytest --run-slow

# Include tests that require network access (disabled by default)
pytest --run-network

# Include tests that interact with exchanges (disabled by default)
pytest --run-exchange
```

### Test Categories

Tests are categorized using markers:

- `@pytest.mark.unit`: Fast, isolated unit tests
- `@pytest.mark.integration`: Tests that interact with multiple components
- `@pytest.mark.slow`: Tests that take a long time to run
- `@pytest.mark.network`: Tests that require network access
- `@pytest.mark.exchange`: Tests that interact with exchanges

## Common Issues and Solutions

### Docstring Style

We use Google style for docstrings. Example of a properly formatted docstring:

```python
def my_function(param1: str, param2: int) -> bool:
    """Do something with parameters.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When parameters are invalid
    """
    return True
```

### Handling Type Checking Errors

If you encounter mypy errors:

1. Check import paths - make sure imports are correct
2. Add type hints to function parameters and return values
3. For third-party libraries without type stubs, add them to `tool.mypy.overrides` in `pyproject.toml`

### Line Length

Maximum line length is 99 characters. Ruff will format code to adhere to this limit, but sometimes manual adjustment is needed.

### Fixing Linting Issues

Ruff can automatically fix many issues:

```bash
# Fix all auto-fixable issues
ruff check --fix .

# Format code
ruff format .
```

### Bypass Pre-commit Hooks

In emergency situations, you can bypass pre-commit hooks with:

```bash
git commit -m "Your message" --no-verify
```

However, this should be used sparingly as the CI pipeline will still run these checks.

## CI/CD Pipeline

Our GitHub Actions workflow runs the same checks as pre-commit hooks, plus additional tests. The workflow is triggered on:

- Pushes to main, master, and develop branches
- Pull requests to these branches
- Manual triggers

To ensure your code passes the CI/CD pipeline, run pre-commit hooks and tests locally before pushing.

## Test Best Practices

1. **Isolate tests**: Each test should focus on testing a single behavior or function
2. **Use fixtures**: Use pytest fixtures for test setup and teardown
3. **Mock external dependencies**: Use pytest-mock to replace external services
4. **Test edge cases**: Include tests for boundary conditions and error cases
5. **Maintain test quality**: Apply the same code quality standards to tests as to production code
6. **Descriptive test names**: Use clear names that describe what the test is verifying
