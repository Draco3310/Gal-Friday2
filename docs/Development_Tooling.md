# Development Tooling Guide

This document outlines the development tools and linting configurations used in the Gal-Friday2 project.

## Configuration Files

The project uses centralized configuration in two primary files:

1. **pyproject.toml** - Contains all tool configurations
2. **.pre-commit-config.yaml** - Defines Git pre-commit hooks that reference configurations in pyproject.toml

## Development Tools

### Code Formatting

- **Black** - Automatic code formatter with consistent style
- **isort** - Import sorting utility configured to be compatible with Black

### Code Quality

- **flake8** - Style guide enforcement
- **pylint** - In-depth static analysis
- **mypy** - Static type checking
- **pydocstyle** - Docstring style checking
- **bandit** - Security vulnerability scanner

### Testing

- **pytest** - Test framework
- **coverage** - Test coverage measurement tool

## Using the Tools

### Manual Execution

You can run these tools manually with the following commands:

```bash
# Formatting
black src tests
isort src tests

# Linting
flake8 src tests
pylint src tests
mypy src tests
pydocstyle src tests
bandit -c pyproject.toml.old src

# Testing
pytest
pytest --cov=src/gal_friday tests/
```

### Pre-commit Hooks

The project uses pre-commit hooks to automatically check your code before committing. To set up:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install
```

After installation, the hooks will run automatically on every commit. You can also run them manually:

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files
pre-commit run
```

## Configuration Details

### Tool-specific Configurations

The `pyproject.toml` file contains specific configurations for each tool, including:

- Line length (99 characters)
- Python version (3.10)
- Ignored files and directories
- Special case exceptions
- Type checking strictness
- Test paths and patterns

### Continuous Integration

The same tools and configurations are used in the CI pipeline to ensure consistency between local development and automated checks.

## Best Practices

1. **Run pre-commit hooks before pushing** to catch issues early
2. **Use the tools as development aids**, not just checkers – many can fix issues automatically
3. **Keep configurations in sync** – all tool settings should live in pyproject.toml
4. **Update tool versions regularly** in the pre-commit config file

## Common Issues

- **Type errors**: If mypy reports errors, check that all function parameters and return values have proper type annotations
- **Import errors**: If isort or flake8 complain about imports, check the ordering and necessity of imports
- **Complexity warnings**: If functions are too complex (C901 errors), consider refactoring them into smaller functions
