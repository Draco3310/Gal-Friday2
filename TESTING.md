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

1. **Black** - Code formatter with a line length of 99 characters
2. **isort** - Import sorter, configured to be compatible with Black
3. **flake8** - Linter for style guide enforcement with plugins for docstring checking
4. **mypy** - Static type checker
5. **pydocstyle** - Docstring style checker (using NumPy convention)
6. **Bandit** - Security issue scanner

## Testing

We use pytest for unit and integration testing. Tests are located in the `tests/` directory.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src tests/

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run with verbose output
pytest -v
```

## Common Issues and Solutions

### Docstring Style

We use NumPy style for docstrings. Example of a properly formatted docstring:

```python
def my_function(param1, param2):
    """Do something with parameters.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int
        Description of param2

    Returns
    -------
    bool
        Description of return value
    """
    return True
```

### Handling Type Checking Errors

If you encounter mypy errors:

1. Check import paths - make sure imports are correct
2. Add type hints to function parameters and return values
3. For third-party libraries without type stubs, add them to `tool.mypy.overrides` in `pyproject.toml`

### Line Length

Maximum line length is 99 characters. Black will format code to adhere to this limit, but sometimes manual adjustment is needed.

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
